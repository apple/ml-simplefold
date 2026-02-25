#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import torch
import numpy as np
from utils.esm_utils import (
    af2_idx_to_esm_idx,
    compute_language_model_representations,
    batch_encode_sequences,
)
from utils.boltz_utils import center_random_augmentation as torch_center_random

try:
    import mlx.core as mx
    from utils.mlx_utils import center_random_augmentation as mlx_center_random
except:
    pass


class ProteinDataProcessor:
    def __init__(
        self, 
        device, 
        scale=16.0, 
        ref_scale=5.0, 
        multiplicity=1,
        backend="torch",
    ):
        self.device = device
        self.scale = scale
        self.ref_scale = ref_scale
        # if multiplicity > 1, effective batch size is multiplicity * batch_size
        self.multiplicity = multiplicity
        self.backend = backend
        if self.backend == "mlx":
            self.center_random_fn = mlx_center_random
        elif self.backend == "torch":
            self.center_random_fn = torch_center_random
        else:
            raise ValueError(f"Unsupported backend: {self.backend}. Choose 'torch' or 'mlx'.")

    def process_esm(
        self, 
        batch, 
        esm_model=None, 
        esm_dict=None, 
        af2_to_esm=None,
        inference=False,
    ):
        sequence = batch["aa_seq"]
        B = len(sequence)
        L = batch["res_type"].shape[1]
        num_tokens = batch["cropped_num_tokens"]

        aatype, mask, residx, linker_mask, _ = batch_encode_sequences(
            sequence, residue_index_offset=512, chain_linker="G" * 25,
        )

        aatype, mask, residx, linker_mask = map(
            lambda x: x.to(self.device), (aatype, mask, residx, linker_mask)
        )

        if residx is None:
            residx = torch.arange(L, device=self.device).expand_as(aatype)

        esmaa = af2_idx_to_esm_idx(aatype, mask, af2_to_esm)

        multiplicity = self.multiplicity if not inference else 1

        esm_s_, _ = compute_language_model_representations(
            esmaa, esm_model, esm_dict, backend=self.backend
        )

        esm_s_ = esm_s_.detach()
        mask, linker_mask = mask.detach().bool(), linker_mask.detach().bool()

        if multiplicity > 1:
            true_mask = linker_mask & mask
            true_len = true_mask[0].sum()
            assert true_len == num_tokens[0]
            esm_s = torch.zeros(
                (1, L, esm_model.num_layers + 1, esm_s_.shape[-1]),
                device=self.device,
            )
            esm_s[0, :true_len] = esm_s_[0, true_mask[0]]
            esm_s = esm_s.repeat_interleave(multiplicity, dim=0)
        else:
            esm_s = torch.zeros(
                (B, L, esm_model.num_layers + 1, esm_s_.shape[-1]),
                device=self.device,
            )
            true_mask = linker_mask & mask
            for i in range(B):
                true_len = true_mask[i].sum()
                assert true_len == num_tokens[i]
                esm_s[i, :true_len] = esm_s_[i, true_mask[i]]

        batch["esm_s"] = esm_s

        return

    def batch_to_device(self, batch, multiplicity=1):
        for k, v in batch.items():
            # if isinstance(v, torch.Tensor) and k in key2cuda:
            if isinstance(v, torch.Tensor):
                if multiplicity > 1:
                    v = v.repeat_interleave(multiplicity, dim=0)
                batch[k] = v.to(self.device)
        return batch

    def batch_to_mlx(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = mx.array(v.numpy())
            if isinstance(v, np.ndarray):
                batch[k] = mx.array(v)
        return batch

    def preprocess_training(self, batch, esm_model=None, esm_dict=None, af2_to_esm=None):
        batch_size, max_ntokens = batch["mol_type"].shape[:2]
        max_natoms = batch["ref_element"].shape[1]

        batch['atom_to_token_idx'] = torch.argmax(
            batch['atom_to_token'], dim=-1)

        y = batch['coords'].float().squeeze(1) / self.scale
        batch['coords'] = y

        ref_y = batch['ref_pos'].float() / self.ref_scale
        batch['ref_pos'] = ref_y

        mol_index = torch.arange(max_natoms).unsqueeze(0).expand(
            batch_size, -1)
        batch['mol_index'] = mol_index

        batch = self.batch_to_device(batch, multiplicity=self.multiplicity)

        if esm_model is not None:
            self.process_esm(batch, esm_model, esm_dict, af2_to_esm)

        # randomly augment the coordinates if repeating batch
        if self.multiplicity > 1:
            batch['coords'] = self.center_random_fn(
                batch['coords'], 
                batch['atom_pad_mask'], 
                centering=True,
                augmentation=True,
            )

        return batch

    def preprocess_inference(self, batch, esm_model=None, esm_dict=None, af2_to_esm=None):
        batch_size, max_ntokens = batch["mol_type"].shape[:2]
        max_natoms = batch["ref_element"].shape[1]

        batch['coords'] = batch['coords'].squeeze(1) / self.scale
        batch['ref_pos'] = batch['ref_pos'].float() / self.ref_scale

        batch['atom_to_token_idx'] = torch.argmax(
            batch['atom_to_token'], dim=-1)

        mol_index = torch.arange(max_natoms).unsqueeze(0).expand(
            batch_size, -1)
        batch['mol_index'] = mol_index

        batch = self.batch_to_device(batch)

        if esm_model is not None and batch.get('esm_s', None) is None:
            print("Processing ESM features for inference...")
            self.process_esm(batch, esm_model, esm_dict, af2_to_esm, inference=True)

        if self.backend == "mlx":
            batch = self.batch_to_mlx(batch)

        return batch

    def postprocess(self, out_dict, batch):
        out_dict['coords'] = self.center_random_fn(
            batch['coords'], 
            batch['atom_pad_mask'], 
            centering=True,
            augmentation=False,
        ) * self.scale
        out_dict['denoised_coords'] = self.center_random_fn(
            out_dict['denoised_coords'], 
            batch['atom_pad_mask'], 
            centering=True,
            augmentation=False,
        ) * self.scale
        return out_dict
