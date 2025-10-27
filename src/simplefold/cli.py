#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import argparse
from simplefold import __version__
from simplefold.inference import predict_structures_from_fastas


def main():
    parser = argparse.ArgumentParser(
        prog="simplefold",
        description="Folding proteins with SimpleFold."
    )
    parser.add_argument("--simplefold_model", type=str, default="simplefold_100M", help="Name of the model to load.")
    parser.add_argument("--ckpt_dir", type=str, default="artifacts", help="Directory to save the checkpoint.")
    parser.add_argument("--output_dir", type=str, default="artifacts/debug_samples", help="Directory to save the output structure.")
    parser.add_argument("--num_steps", type=int, default=500, help="Number of steps in inference.")
    parser.add_argument("--tau", type=float, default=0.1, help="Diffusion coefficient scaling factor.")
    parser.add_argument("--no_log_timesteps", action="store_true", help="Disable logarithmic timesteps.")
    parser.add_argument("--fasta_path", required=True, type=str, help="Path to the input FASTA file/directory.")
    parser.add_argument("--nsample_per_protein", type=int, default=1, help="Number of samples to generate per protein.")
    parser.add_argument("--plddt", action="store_true", help="Enable pLDDT prediction.")
    parser.add_argument("--output_format", type=str, default="mmcif", choices=["pdb", "mmcif"], help="Output file format.")
    parser.add_argument("--backend", type=str, default='torch', choices=['torch', 'mlx'], help="Backend to run inference either torch or mlx")
    parser.add_argument("--cache", type=str, default=None, help="Specify the cache directory other than default.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    args = parser.parse_args()

    print(f"Running protein folding with SimpleFold ...")
    predict_structures_from_fastas(args)
