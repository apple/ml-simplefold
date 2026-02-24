#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import os
import shutil
import copy
import numpy as np
from pathlib import Path
from einops import repeat

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.optim.swa_utils import AveragedModel
from torch.nn.utils import clip_grad_norm_

import lightning
import lightning.pytorch as pl
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap

from utils.esm_utils import _af2_to_esm, esm_registry
from boltz_data_pipeline.types import Record, Structure
from utils.boltz_utils import (
    weighted_rigid_align, 
    center_random_augmentation,
    process_structure, 
    save_structure
)


def logit_normal_sample(n=1, m=0.0, s=1.0):
    # Logit-Normal Sampling from https://arxiv.org/pdf/2403.03206.pdf
    u = torch.randn(n) * s + m
    t = 1 / (1 + torch.exp(-u))
    return t


def lddt_dist(dmat_predicted, dmat_true, mask, cutoff=15.0, per_atom=False):
    # NOTE: the mask is a pairwise mask which should have the identity elements already masked out
    # Compute mask over distances
    dists_to_score = (dmat_true < cutoff).float() * mask
    dist_l1 = torch.abs(dmat_true - dmat_predicted)

    score = 0.25 * (
        (dist_l1 < 0.5).float()
        + (dist_l1 < 1.0).float()
        + (dist_l1 < 2.0).float()
        + (dist_l1 < 4.0).float()
    )

    # Normalize over the appropriate axes.
    if per_atom:
        mask_no_match = torch.sum(dists_to_score, dim=-1) != 0
        norm = 1.0 / (1e-10 + torch.sum(dists_to_score, dim=-1))
        score = norm * (1e-10 + torch.sum(dists_to_score * score, dim=-1))
        return score, mask_no_match.float()
    else:
        norm = 1.0 / (1e-10 + torch.sum(dists_to_score, dim=(-2, -1)))
        score = norm * (1e-10 + torch.sum(dists_to_score * score, dim=(-2, -1)))
        total = torch.sum(dists_to_score, dim=(-1, -2))
        return score, total


class SimpleFold(pl.LightningModule):
    def __init__(
        self,
        architecture,
        processor,
        loss,
        path,
        sampler,
        optimizer=None,
        scheduler=None,
        plddt_module=None,
        ema_decay=0.999,
        esm_model="esm2_3B",
        aa_bolt_link=None,
        use_rigid_align=True,
        smooth_lddt_loss_weight=1.0,
        lddt_cutoff=15.0,
        clip_grad_norm_val=None,
        lddt_weight_schedule=False,
        plddt_training=False,
        sample_dir='artifacts/',
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = architecture
        self.model_ema = AveragedModel(
            self.model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(
                self.hparams.ema_decay
            ),
            use_buffers=True,
        )
        self.model_ema.eval()

        self.loss = loss
        self.path = path
        self.sampler = sampler

        self.use_rigid_align = use_rigid_align
        self.lddt_cutoff = lddt_cutoff
        self.smooth_lddt_loss_weight = smooth_lddt_loss_weight
        self.use_smooth_lddt_loss = smooth_lddt_loss_weight > 0.0
        self.lddt_weight_schedule = lddt_weight_schedule
        self.plddt_training = plddt_training
        self.sample_dir = sample_dir

        self.aa_bolt_link = aa_bolt_link
        self.nval_steps = 0

        try:
            self.t_eps = self.sampler.t_eps
        except AttributeError:
            self.t_eps = 0.0

        self.use_esm = esm_model is not None
        if self.use_esm:
            self.esm_model, self.esm_dict = esm_registry[esm_model]()
            self.esm_model.eval()
            self.af2_to_esm = _af2_to_esm(self.esm_dict)
            print(f"Using ESM model: {esm_model}")
        else:
            self.esm_model = None
            self.esm_dict = None
            self.af2_to_esm = None

        self.plddt_module = plddt_module
        if self.plddt_training:
            assert self.plddt_module is not None, "PLDDT module must be provided for PLDDT training"
            self.model.eval()

    def register(self, name, tensor):
        self.register_buffer(name, tensor.type(torch.float32))

    def loss_masking(self, loss, atom_mask):
        loss_mask = repeat(atom_mask, "b s -> b s d", d=loss.shape[-1])
        loss *= loss_mask

        denom = torch.sum(atom_mask, -1, keepdim=True)
        denom = denom.unsqueeze(-1)
        loss = torch.sum(loss, dim=1, keepdim=True) / denom
        return loss

    def smooth_lddt_loss(
        self, 
        pred_coords,
        true_coords,
        # is_nucleotide,
        coords_mask,
        t,
    ):
        """Compute weighted alignment.

        Parameters
        ----------
        pred_coords: torch.Tensor
            The predicted atom coordinates
        true_coords: torch.Tensor
            The ground truth atom coordinates
        coords_mask: torch.Tensor
            The atoms mask

        """
        B, N, _ = true_coords.shape
        true_dists = torch.cdist(true_coords, true_coords)

        mask = (true_dists < self.lddt_cutoff).float()
        mask = mask * (1 - torch.eye(pred_coords.shape[1], device=pred_coords.device))
        mask = mask * (coords_mask.unsqueeze(-1) * coords_mask.unsqueeze(-2))

        # Compute distances between all pairs of atoms
        pred_dists = torch.cdist(pred_coords, pred_coords)
        dist_diff = torch.abs(true_dists - pred_dists)

        # Compute epsilon values
        eps = (
            (
                (
                    F.sigmoid(0.5 - dist_diff)
                    + F.sigmoid(1.0 - dist_diff)
                    + F.sigmoid(2.0 - dist_diff)
                    + F.sigmoid(4.0 - dist_diff)
                )
                / 4.0
            )
        )

        # Calculate masked averaging
        num = (eps * mask).sum(dim=(-1, -2))
        den = mask.sum(dim=(-1, -2)).clamp(min=1)
        lddt = num / den
        if self.lddt_weight_schedule:
            t_weight = 1 + 8 * torch.relu(t - 0.5)
            lddt = (1.0 - lddt) * t_weight
            return lddt.mean()
        else:
            return (1.0 - lddt.mean()) * self.smooth_lddt_loss_weight

    def plddt_loss(
        self,
        pred_lddt,
        pred_atom_coords,
        true_atom_coords,
        true_coords_resolved_mask,
        feats,
        # multiplicity=1,
    ):
        """Compute plddt loss.

        Parameters
        ----------
        pred_lddt: torch.Tensor
            The plddt logits
        pred_atom_coords: torch.Tensor
            The predicted atom coordinates
        true_atom_coords: torch.Tensor
            The atom coordinates after symmetry correction
        true_coords_resolved_mask: torch.Tensor
            The resolved mask after symmetry correction
        feats: Dict[str, torch.Tensor]
            Dictionary containing the model input

        Returns
        -------
        torch.Tensor
            Plddt loss

        """

        # extract necessary features
        atom_mask = true_coords_resolved_mask

        R_set_to_rep_atom = feats["r_set_to_rep_atom"].float()
        # R_set_to_rep_atom = R_set_to_rep_atom.repeat_interleave(multiplicity, 0).float()

        token_type = feats["mol_type"]
        # token_type = token_type.repeat_interleave(multiplicity, 0)
        # is_nucleotide_token = (token_type == const.chain_type_ids["DNA"]).float() + (
        #     token_type == const.chain_type_ids["RNA"]
        # ).float()

        B = true_atom_coords.shape[0]

        # atom_to_token = feats["atom_to_token"].float()
        # atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)

        token_to_rep_atom = feats["token_to_rep_atom"].float()
        # token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0)

        true_token_coords = torch.bmm(token_to_rep_atom, true_atom_coords)
        pred_token_coords = torch.bmm(token_to_rep_atom, pred_atom_coords)

        # compute true lddt
        true_d = torch.cdist(
            true_token_coords,
            torch.bmm(R_set_to_rep_atom, true_atom_coords),
        )
        pred_d = torch.cdist(
            pred_token_coords,
            torch.bmm(R_set_to_rep_atom, pred_atom_coords),
        )

        # compute mask
        pair_mask = atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2)
        pair_mask = (
            pair_mask
            * (1 - torch.eye(pair_mask.shape[1], device=pair_mask.device))[None, :, :]
        )
        pair_mask = torch.einsum("bnm,bkm->bnk", pair_mask, R_set_to_rep_atom)
        pair_mask = torch.bmm(token_to_rep_atom, pair_mask)
        atom_mask = torch.bmm(token_to_rep_atom, atom_mask.unsqueeze(-1).float())
        # is_nucleotide_R_element = torch.bmm(
        #     R_set_to_rep_atom, torch.bmm(atom_to_token, is_nucleotide_token.unsqueeze(-1))
        # ).squeeze(-1)
        # cutoff = 15 + 15 * is_nucleotide_R_element.reshape(B, 1, -1).repeat(
        #     1, true_d.shape[1], 1
        # )

        # compute lddt
        target_lddt, mask_no_match = lddt_dist(
            pred_d, true_d, pair_mask, cutoff=15.0, per_atom=True
        )

        # compute loss
        num_bins = pred_lddt.shape[-1]
        bin_index = torch.floor(target_lddt * num_bins).long()
        bin_index = torch.clamp(bin_index, max=(num_bins - 1))
        lddt_one_hot = F.one_hot(bin_index, num_classes=num_bins)
        errors = -1 * torch.sum(
            lddt_one_hot * F.log_softmax(pred_lddt, dim=-1),
            dim=-1,
        )
        atom_mask = atom_mask.squeeze(-1)
        loss = torch.sum(errors * atom_mask * mask_no_match, dim=-1) / (
            1e-7 + torch.sum(atom_mask * mask_no_match, dim=-1)
        )

        # Average over the batch dimension
        loss = torch.mean(loss)

        self.log(
            "loss/plddt",
            loss.item(),
            on_epoch=True,
            logger=True,
            prog_bar=True,
            rank_zero_only=True,
        )

        return loss

    def plddt_train_step(self, batch, batch_idx):
        with torch.no_grad():
            batch = self.processor.preprocess_training(
                batch,
                esm_model=self.esm_model,
                esm_dict=self.esm_dict,
                af2_to_esm=self.af2_to_esm,
            )

            noise = torch.randn_like(batch['coords']).to(self.device)

            out_dict = self.sampler.sample(
                self.model_ema.module.forward, self.path,
                noise, batch
            )
            # out_dict = self.processor.postprocess(out_dict, batch)

            # denoised_coords = center_of_mass_norm(
            #     out_dict["denoised_coords"], batch['atom_pad_mask']
            # )
            # true_coords = center_of_mass_norm(
            #     batch['coords'], batch['atom_pad_mask']
            # )
            denoised_coords = center_random_augmentation(
                out_dict["denoised_coords"],
                batch['atom_pad_mask'],
                augmentation=False,
                centering=True,
            )
            true_coords = center_random_augmentation(
                batch['coords'],
                batch['atom_pad_mask'],
                augmentation=False,
                centering=True,
            )
            out_dict["denoised_coords"] = denoised_coords * self.processor.scale
            out_dict["coords"] = true_coords * self.processor.scale

            out_dict["true_coords_resolved_mask"] = batch["atom_resolved_mask"]

            t = torch.ones(batch['coords'].shape[0], device=self.device)
            out_feat = self.model(denoised_coords, t, batch) # use unscaled coords

        # Compute plddt loss
        plddt_out_dict = self.plddt_module(
            out_feat["latent"].detach(),
            batch,
        )

        plddt_loss = self.plddt_loss(
            plddt_out_dict["plddt_logits"],
            out_dict["denoised_coords"],
            out_dict["coords"],
            out_dict["true_coords_resolved_mask"],
            batch,
        )

        return plddt_loss

    def flow_matching_train_step(self, batch, batch_idx):
        batch = self.processor.preprocess_training(
            batch,
            esm_model=self.esm_model,
            esm_dict=self.esm_dict,
            af2_to_esm=self.af2_to_esm,
        )

        # timestep resampling
        t_size = batch['coords'].shape[0]
        t = 0.98 * logit_normal_sample(n=t_size, m=0.8, s=1.7) + 0.02 * torch.rand(t_size)
        t = t.to(self.device)
        t = t * (1 - 2 * self.t_eps) + self.t_eps

        noise = torch.randn_like(batch['coords']).to(self.device)

        _, y_t, v_t = self.path.interpolant(t, noise, batch["coords"])

        out_dict = self.model(y_t, t, batch)

        resolved_atom_mask = batch["atom_resolved_mask"].float()
        align_weights = y_t.new_ones(y_t.shape[:2])

        if self.use_rigid_align:
            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                v_t = out_dict['predict_velocity'].detach().float()
                denoised_coords = y_t + v_t * (1.0 - t[:, None, None])
                coords = batch["coords"].detach().float()
                coords_aligned = weighted_rigid_align(
                    coords,
                    denoised_coords.detach().float(),
                    align_weights.detach().float(),
                    mask=resolved_atom_mask.detach().float(),
                )
                _, _, v_t_aligned = self.path.interpolant(t, noise, coords_aligned)
            target = v_t_aligned
        else:
            target = v_t

        loss = F.mse_loss(out_dict['predict_velocity'], target, reduction='none')

        loss_mask = resolved_atom_mask * align_weights
        loss = self.loss_masking(loss, loss_mask)
        loss = loss.mean()

        self.log(
            "loss/mse",
            loss.item(),
            on_epoch=True,
            logger=True,
            prog_bar=True,
            rank_zero_only=True,
        )

        if self.use_smooth_lddt_loss:
            # one-step Euler to get denoised coordinates
            denoised_coords = y_t + \
                out_dict['predict_velocity'] * (1.0 - t[:, None, None])

            # rescale coordinates to angstroms
            # denoised_coords = center_of_mass_norm(denoised_coords, batch['atom_pad_mask'])
            # true_coords = center_of_mass_norm(batch['coords'], batch['atom_pad_mask'])
            denoised_coords = center_random_augmentation(
                denoised_coords,
                batch['atom_pad_mask'],
                augmentation=False,
                centering=True,
            )
            true_coords = center_random_augmentation(
                batch['coords'],
                batch['atom_pad_mask'],
                augmentation=False,
                centering=True,
            )
            denoised_coords = denoised_coords * self.processor.scale
            true_coords = true_coords * self.processor.scale

            smooth_lddt_loss = self.smooth_lddt_loss(
                denoised_coords,
                true_coords,
                resolved_atom_mask,
                t,
            )
            loss += smooth_lddt_loss

            self.log(
                "loss/smooth_lddt",
                smooth_lddt_loss.item(),
                on_epoch=True,
                logger=True,
                prog_bar=True,
                rank_zero_only=True,
            )

        self.log(
            "loss/loss",
            loss.item(),
            on_epoch=True,
            logger=True,
            prog_bar=True,
            rank_zero_only=True,
        )

        self.log(
            "trainer/global_step",
            self.global_step,
            on_epoch=False,
            logger=True,
            prog_bar=False,
            rank_zero_only=True,
        )

        self.global_training_step = self.trainer.global_step
        self.epoch = self.trainer.current_epoch
        self.world_size = self.trainer.world_size
        return loss

    def training_step(self, batch, batch_idx):
        if self.plddt_training:
            return self.plddt_train_step(batch, batch_idx)
        else:
            return self.flow_matching_train_step(batch, batch_idx)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # we skip validation step in training mode
        return

    def on_train_start(self):
        global_seed = os.environ.get("PL_GLOBAL_SEED", 42)
        pl.seed_everything(int(global_seed) + self.trainer.global_rank, True)
        return

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            batch = self.processor.preprocess_inference(
                batch,
                esm_model=self.esm_model,
                esm_dict=self.esm_dict,
                af2_to_esm=self.af2_to_esm,
            )
            num_repeats = batch.get("num_repeats", torch.tensor(1, device=self.device)).item()
            multiplicity = batch["mol_type"].shape[0]
            num_iter = np.ceil(num_repeats / multiplicity).astype(int)
            print(f"Generating {num_repeats} samples with num_iter: {num_iter}, multiplicity: {multiplicity}")

            # num_repeats is the total number of samples to generate for one protein
            # multiplicity is the number of samples to generate at once

            curr_idx = 0
            for i in range(num_iter):
                batch_in = copy.deepcopy(batch)
                noise = torch.randn_like(batch_in['coords']).to(self.device)

                out_dict = self.sampler.sample(
                    self.model_ema.module.forward, self.path,
                    noise, batch_in
                )

                if self.plddt_module is not None:
                    denoised_coords = center_random_augmentation(
                        out_dict["denoised_coords"],
                        batch['atom_pad_mask'],
                        augmentation=False,
                        centering=True,
                    )

                    t = torch.ones(batch['coords'].shape[0], device=self.device)
                    out_feat = self.model(denoised_coords, t, batch) # use unscaled coords

                    plddt_out_dict = self.plddt_module(
                        out_feat["latent"],
                        batch_in,
                    )
                    plddts = plddt_out_dict["plddt"] * 100.0
                else:
                    plddts = None

                out_dict = self.processor.postprocess(out_dict, batch_in)

                record = Record(**batch_in['record'][0])
                gt_coord = out_dict['coords']
                sampled_coord = out_dict['denoised_coords']
                pad_mask = batch_in['atom_pad_mask']

                if num_repeats - curr_idx < multiplicity:
                    curr_num_copies = num_repeats - curr_idx
                    gt_coord = gt_coord[:curr_num_copies]
                    sampled_coord = sampled_coord[:curr_num_copies]
                    pad_mask = pad_mask[:curr_num_copies]
                    plddts = plddts[:curr_num_copies] if plddts is not None else None
                else:
                    curr_num_copies = multiplicity

                data_dir = self.trainer.datamodule.predict_dataloader().dataset.target_dir
                sample_dir = self.sample_dir
                path = Path(data_dir) / "structures" / f"{record.id}.npz"
                structure: Structure = Structure.load(path)

                for j in range(curr_num_copies):
                    file_id = j + curr_idx
                    try:
                        sampled_structure = copy.deepcopy(structure)
                        sampled_structure = process_structure(
                            sampled_structure, sampled_coord[j], pad_mask[j], record
                        )
                        # save mmcif structure
                        sampled_struct_dir = Path(sample_dir)
                        outname = f"{record.id}_sampled_{str(file_id)}"
                        save_structure(
                            sampled_structure, sampled_struct_dir, outname, 
                            plddts=plddts[j] if plddts is not None else None,
                            output_format="mmcif",
                        )
                        # save pdb structure
                        save_structure(
                            sampled_structure, sampled_struct_dir, outname, 
                            plddts=plddts[j] if plddts is not None else None,
                            output_format="pdb",
                        )
                    except:
                        print(f"Error processing {record.id}")
                        continue

                curr_idx += curr_num_copies

            self.world_size = self.trainer.world_size

        return

    def on_predict_epoch_end(self):
        dist.barrier() # wait for all processes to finish
        return

    def reset_esm(self, esm_model: str):
        self.esm_model, self.esm_dict = esm_registry[esm_model]()
        self.esm_model.eval()
        self.af2_to_esm = _af2_to_esm(self.esm_dict)
        self.esm_model = self.esm_model.to(self.device)
        self.af2_to_esm = self.af2_to_esm.to(self.device)
        print(f"Successfully reset ESM model {esm_model}")

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        self.processor = self.hparams.processor(device=self.device)

        if self.use_esm:
            if stage == "fit" and not isinstance(
                self.trainer.strategy, lightning.pytorch.strategies.fsdp.FSDPStrategy
            ):
                # initialize the model with FSDP wrapper
                fsdp_params = dict(
                    mixed_precision=True,
                    flatten_parameters=True,
                    state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
                    cpu_offload=True,  # enable cpu offloading
                    fp32_reduce_scatter=True,  # use fp32 reduce scatter
                )
                with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
                    self.esm_model.eval()
                    # Wrap each layer in FSDP separately
                    for name, child in self.esm_model.named_children():
                        if name == "layers":
                            for layer_name, layer in child.named_children():
                                wrapped_layer = wrap(layer)
                                setattr(child, layer_name, wrapped_layer)
                    self.esm_model = wrap(self.esm_model)
                self.af2_to_esm = self.af2_to_esm.to(self.device)
            else:
                self.esm_model = self.esm_model.to(self.device)
                self.af2_to_esm = self.af2_to_esm.to(self.device)

        if stage == "fit":
            self.training_gpus = self.trainer.world_size
            self.hparams["training_gpus"] = self.training_gpus

            batch = next(iter(self.trainer.datamodule.train_dataloader()))

            batch = self.processor.preprocess_training(
                batch,
                self.esm_model,
                self.esm_dict,
                self.af2_to_esm,
            )
            y = batch["coords"]
            t = torch.zeros((y.shape[0])).cuda()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        optimizer = self.optimizers()
        self.log(
            "trainer/lr",
            optimizer.param_groups[0]["lr"],
            on_epoch=True,
            logger=True,
            prog_bar=True,
            rank_zero_only=True,
        )

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:

        if isinstance(
            self.trainer.strategy, lightning.pytorch.strategies.fsdp.FSDPStrategy
        ):

            with FullyShardedDataParallel.summon_full_params(
                self.trainer.strategy.model, with_grads=True
            ):
                clip_grad_norm_(
                    self.trainer.strategy.model.model.parameters(),
                    self.hparams.clip_grad_norm_val,
                    norm_type=2.0,
                    error_if_nonfinite=True,
                )

        else:
            clip_grad_norm_(
                self.model.parameters(),
                self.hparams.clip_grad_norm_val,
                norm_type=2.0,
                error_if_nonfinite=True,
            )
        return

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        # if self.eval_ema:
        if isinstance(
            self.trainer.strategy, lightning.pytorch.strategies.fsdp.FSDPStrategy
        ):

            with FullyShardedDataParallel.summon_full_params(
                self.trainer.strategy.model
            ):
                self.trainer.strategy.model.model_ema.update_parameters(
                    self.trainer.strategy.model.model
                )

        else:
            self.model_ema.update_parameters(self.model)
        return

    def on_save_checkpoint(self, checkpoint) -> None:

        if not isinstance(
            self.trainer.strategy, lightning.pytorch.strategies.fsdp.FSDPStrategy
        ):
            layers_to_delete = []
            for k in checkpoint["state_dict"].keys():
                if k.startswith("esm_model._fsdp_wrapped_module"):
                    layers_to_delete.append(k)

            for k in layers_to_delete:
                del checkpoint["state_dict"][k]

        else:
            try:
                del checkpoint["hyper_parameters"]
            except:
                print("No hyper_parameters in checkpoint")

        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint) -> None:

        if not isinstance(
            self.trainer.strategy, lightning.pytorch.strategies.fsdp.FSDPStrategy
        ):
            self.training_gpus = checkpoint["hyper_parameters"]["training_gpus"]
            self.fwd_flops = checkpoint["hyper_parameters"]["fwd_flops"]

        if checkpoint["loops"] is not None:

            self.trainer.fit_loop.load_state_dict(checkpoint["loops"]["fit_loop"])
            self.trainer.validate_loop.load_state_dict(
                checkpoint["loops"]["validate_loop"]
            )
        return super().on_load_checkpoint(checkpoint)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(
            params=self.trainer.model.parameters()
        )

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
