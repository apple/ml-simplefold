#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import os
import torch
import hydra
import omegaconf
from pathlib import Path
from itertools import starmap

from model.flow import LinearPath
from model.torch.sampler import EMSampler

from processor.protein_processor import ProteinDataProcessor
from utils.datamodule_utils import process_one_inference_structure
from utils.esm_utils import _af2_to_esm, esm_registry
from utils.boltz_utils import process_structure, save_structure
from utils.fasta_utils import process_fastas, download_fasta_utilities
from boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer

try: 
    import mlx.core as mx
    from mlx.utils import tree_unflatten, tree_flatten
    from model.mlx.sampler import EMSampler as EMSamplerMLX
    from model.mlx.esm_network import ESM2 as ESM2MLX
    from utils.mlx_utils import map_torch_to_mlx, map_plddt_torch_to_mlx
    MLX_AVAILABLE = True
except:
    MLX_AVAILABLE = False
    print("MLX not installed, skip importing MLX related packages.")


ckpt_url_dict = {
    "simplefold_100M": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_100M.ckpt",
    "simplefold_360M": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_360M.ckpt",
    "simplefold_700M": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_700M.ckpt",
    "simplefold_1.1B": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_1.1B.ckpt",
    "simplefold_1.6B": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_1.6B.ckpt",
    "simplefold_3B": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_3B.ckpt",
}

plddt_ckpt_url = (
    "https://ml-site.cdn-apple.com/models/simplefold/plddt_module_1.6B.ckpt"
)


class ModelWrapper:
    def __init__(
        self,
        simplefold_model,
        plddt=False,
        ckpt_dir="./artifacts",
        backend="torch",
    ):
        self.simplefold_model = simplefold_model
        self.plddt = plddt
        self.ckpt_dir = Path(ckpt_dir)
        self.backend = backend
        if self.backend == "mlx" and not MLX_AVAILABLE:
            self.backend = "torch"
            print("MLX not installed, switch to torch backend.")
        self.folding_model = None
        self.plddt_out_module = None
        self.plddt_latent_module = None
        self.device = self.get_device()
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def get_device(self):
        if self.backend == "torch":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.backend == "mlx":
            device = "cpu"
        return device

    def from_pretrained_folding_model(self):
        # define folding model
        simplefold_model = self.simplefold_model

        # create folding model
        ckpt_path = os.path.join(self.ckpt_dir, f"{simplefold_model}.ckpt")
        if not os.path.exists(ckpt_path):
            os.system(f"curl -L -o {ckpt_path} {ckpt_url_dict[simplefold_model]}")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # load model checkpoint
        cfg_path = os.path.join(
            "configs/model/architecture", f"foldingdit_{simplefold_model[11:]}.yaml"
        )
        if self.backend == "torch":
            model_config = omegaconf.OmegaConf.load(cfg_path)
            model = hydra.utils.instantiate(model_config)
            model.load_state_dict(checkpoint, strict=True)
            model = model.to(self.device)
        elif self.backend == "mlx":
            # replace torch implementations with mlx
            with open(cfg_path, "r") as f:
                yaml_str = f.read()
            yaml_str = yaml_str.replace("torch", "mlx")

            model_config = omegaconf.OmegaConf.create(yaml_str)
            model = hydra.utils.instantiate(model_config)
            mlx_state_dict = {
                k: mx.array(v)
                for k, v in starmap(map_torch_to_mlx, checkpoint.items())
                if k is not None
            }
            model.update(tree_unflatten(list(mlx_state_dict.items())))
        print(f"Folding model {simplefold_model} loaded with {self.backend} backend.")

        model.eval()
        return model

    def from_pretrained_plddt_model(self):
        if not self.plddt:
            return {
                "plddt_out_module": None,
                "plddt_latent_module": None,
            }

        # load pLDDT module if specified
        plddt_ckpt_path = os.path.join(self.ckpt_dir, "plddt.ckpt")
        if not os.path.exists(plddt_ckpt_path):
            os.system(f"curl -L -o {plddt_ckpt_path} {plddt_ckpt_url}")

        plddt_module_path = "configs/model/architecture/plddt_module.yaml"
        plddt_checkpoint = torch.load(
            plddt_ckpt_path, map_location="cpu", weights_only=False
        )

        if self.backend == "torch":
            plddt_config = omegaconf.OmegaConf.load(plddt_module_path)
            plddt_out_module = hydra.utils.instantiate(plddt_config)
            plddt_out_module.load_state_dict(plddt_checkpoint, strict=True)
            plddt_out_module = plddt_out_module.to(self.device)
        elif self.backend == "mlx":
            # replace torch implementations with mlx
            with open(plddt_module_path, "r") as f:
                yaml_str = f.read()
            yaml_str = yaml_str.replace("torch", "mlx")

            plddt_config = omegaconf.OmegaConf.create(yaml_str)
            plddt_out_module = hydra.utils.instantiate(plddt_config)

            mlx_state_dict = {
                k: mx.array(v)
                for k, v in starmap(map_plddt_torch_to_mlx, plddt_checkpoint.items())
                if k is not None
            }
            plddt_out_module.update(tree_unflatten(list(mlx_state_dict.items())))

        plddt_out_module.eval()
        print(f"pLDDT output module loaded with {self.backend} backend.")

        plddt_latent_ckpt_path = os.path.join(self.ckpt_dir, "simplefold_1.6B.ckpt")
        if not os.path.exists(plddt_latent_ckpt_path):
            os.makedirs(self.ckpt_dir, exist_ok=True)
            os.system(
                f"curl -L -o {plddt_latent_ckpt_path} {ckpt_url_dict['simplefold_1.6B']}"
            )

        plddt_latent_config_path = "configs/model/architecture/foldingdit_1.6B.yaml"
        plddt_latent_checkpoint = torch.load(
            plddt_latent_ckpt_path, map_location="cpu", weights_only=False
        )

        if self.backend == "torch":
            plddt_latent_config = omegaconf.OmegaConf.load(plddt_latent_config_path)
            plddt_latent_module = hydra.utils.instantiate(plddt_latent_config)
            plddt_latent_module.load_state_dict(plddt_latent_checkpoint, strict=True)
            plddt_latent_module = plddt_latent_module.to(self.device)
        elif self.backend == "mlx":
            # replace torch implementations with mlx
            with open(plddt_latent_config_path, "r") as f:
                yaml_str = f.read()
            yaml_str = yaml_str.replace("torch", "mlx")

            plddt_latent_config = omegaconf.OmegaConf.create(yaml_str)
            plddt_latent_module = hydra.utils.instantiate(plddt_latent_config)
            mlx_state_dict = {
                k: mx.array(v)
                for k, v in starmap(map_torch_to_mlx, plddt_latent_checkpoint.items())
                if k is not None
            }
            plddt_latent_module.update(tree_unflatten(list(mlx_state_dict.items())))

        plddt_latent_module.eval()
        print(f"pLDDT latent module loaded with {self.backend} backend.")

        return {
            "plddt_out_module": plddt_out_module,
            "plddt_latent_module": plddt_latent_module,
        }


class InferenceWrapper:
    def __init__(
        self,
        output_dir,
        prediction_dir,
        num_steps,
        nsample_per_protein,
        tau,
        device,
        backend,
    ):
        self.num_steps = num_steps
        self.nsample_per_protein = nsample_per_protein
        self.tau = tau
        self.device = device
        self.backend = backend

        if self.backend == "mlx" and not MLX_AVAILABLE:
            self.backend = "torch"
            print("MLX not installed, switch to torch backend.")

        # create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # create cache directory
        cache = output_dir / "cache"
        cache.mkdir(parents=True, exist_ok=True)

        # create prediction directory
        prediction_dir = output_dir / prediction_dir
        prediction_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir = output_dir
        self.cache = cache
        self.prediction_dir = prediction_dir

        self.initialize_esm_model()
        self.initialize_others()

    def initialize_esm_model(self):
        # load ESM2 model
        esm_model, esm_dict = esm_registry["esm2_3B"]()
        af2_to_esm = _af2_to_esm(esm_dict)

        if self.backend == "torch":
            esm_model = esm_model.to(self.device)
            af2_to_esm = af2_to_esm.to(self.device)
        elif self.backend == "mlx":
            esm_model_mlx = ESM2MLX(num_layers=36, embed_dim=2560, attention_heads=40)
            esm_state_dict_torch = esm_model.cpu().state_dict()

            esm_state_dict_torch = {
                k: mx.array(v)
                for k, v in starmap(map_torch_to_mlx, esm_state_dict_torch.items())
                if k is not None
            }
            esm_model_mlx.update(tree_unflatten(list(esm_state_dict_torch.items())))
            esm_model = esm_model_mlx
        print(f"pLM ESM-3B loaded with {self.backend} backend.")

        self.esm_model = esm_model.eval()
        self.esm_dict = esm_dict
        self.af2_to_esm = af2_to_esm

    def initialize_others(self):
        # prepare data tokenizer, featurizer, and processor
        self.tokenizer = BoltzTokenizer()
        self.featurizer = BoltzFeaturizer()
        self.processor = ProteinDataProcessor(
            device=self.device,
            scale=16.0,
            ref_scale=5.0,
            multiplicity=1,
            backend=self.backend,
        )

        # define flow process and sampler
        self.flow = LinearPath()

        if self.backend == "torch":
            sampler_cls = EMSampler
        elif self.backend == "mlx":
            sampler_cls = EMSamplerMLX

        self.sampler = sampler_cls(
            num_timesteps=self.num_steps,
            t_start=1e-4,
            tau=self.tau,
            log_timesteps=True,
            w_cutoff=0.99,
        )

    def process_input(self, aa_seq):
        # process fasta files to input format
        download_fasta_utilities(self.cache)
        # save the input sequence to a fasta file
        with open(self.cache / "input.fasta", "w") as f:
            f.write(f">A|Protein\n{aa_seq}\n")
        data = [self.cache / "input.fasta"]
        process_fastas(
            data=data,
            out_dir=self.cache,
            ccd_path=self.cache / "ccd.pkl",
        )

        # prepare the target protein data for inference
        struct_file = self.cache / "structures" / "input.npz"
        record_file = self.cache / "records" / "input.json"
        batch, structure, record = process_one_inference_structure(
            struct_file,
            record_file,
            self.tokenizer,
            self.featurizer,
            self.processor,
            self.esm_model,
            self.esm_dict,
            self.af2_to_esm,
        )
        return batch, structure, record

    def run_inference(self, batch, model, plddt_model, device):
        # run inference for target protein
        plddt_out_module = plddt_model["plddt_out_module"]
        plddt_latent_module = plddt_model["plddt_latent_module"]
        coord_samples = []
        if plddt_latent_module is None or plddt_out_module is None:
            compute_plddt = False
            plddt_samples = None
        else:
            compute_plddt = True
            plddt_samples = []
        pad_mask = batch["atom_pad_mask"]
        plddts = None

        for _ in range(self.nsample_per_protein):
            if self.backend == "torch":
                noise = torch.randn_like(batch["coords"]).to(device)
            elif self.backend == "mlx":
                noise = mx.random.normal(batch["coords"].shape)
            out_dict = self.sampler.sample(model, self.flow, noise, batch)

            if compute_plddt:
                if self.backend == "torch":
                    t = torch.ones(batch["coords"].shape[0], device=device)
                    out_feat = plddt_latent_module(
                        out_dict["denoised_coords"].detach(), t, batch
                    )
                    plddt_out_dict = plddt_out_module(
                        out_feat["latent"].detach(),
                        batch,
                    )
                elif self.backend == "mlx":
                    t = mx.ones(batch["coords"].shape[0])
                    out_feat = plddt_latent_module(out_dict["denoised_coords"], t, batch)
                    plddt_out_dict = plddt_out_module(
                        out_feat["latent"],
                        batch,
                    )
                # scale pLDDT to [0, 100]
                plddt_samples.append(plddt_out_dict["plddt"] * 100.0)

            out_dict = self.processor.postprocess(out_dict, batch)
            if self.backend == "torch":
                coord_samples.append(out_dict["denoised_coords"].detach())
            else:
                coord_samples.append(out_dict["denoised_coords"])

        if self.backend == "torch":
            sampled_coord = torch.cat(coord_samples, dim=0)
            pad_mask = pad_mask.detach().repeat_interleave(
                self.nsample_per_protein, dim=0
            )
            if compute_plddt:
                plddts = torch.cat(plddt_samples, dim=0).detach()
        else:
            sampled_coord = mx.concatenate(coord_samples, axis=0)
            pad_mask = mx.concatenate(
                [pad_mask] * self.nsample_per_protein, axis=0
            )
            if compute_plddt:
                plddts = mx.concatenate(plddt_samples, axis=0)

        return {
            "sampled_coord": sampled_coord,
            "pad_mask": pad_mask,
            "plddts": plddts,
        }

    def save_result(self, structure, record, results, out_name):
        sampled_coord = results["sampled_coord"]
        pad_mask = results["pad_mask"]
        plddt = results["plddts"]

        save_paths = []
        for i in range(sampled_coord.shape[0]):
            sampled_coord_i = sampled_coord[i]
            pad_mask_i = pad_mask[i]
            plddt_i = plddt[i] if plddt is not None else None
            out_name_i = f"{out_name}_sampled_{i}"
            # save the generated structure
            structure_save = process_structure(
                structure, sampled_coord_i, pad_mask_i, record, backend=self.backend
            )
            save_structure(
                structure_save,
                self.prediction_dir,
                out_name_i,
                output_format="mmcif",
                plddts=plddt_i,
            )
            save_paths.append(self.prediction_dir / f"{out_name_i}.cif")
        return save_paths
