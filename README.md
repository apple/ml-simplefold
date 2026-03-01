
<h1 align="center"><strong>SimpleFold: Folding Proteins is Simpler than You Think</strong></h1>


<div align="center">

This github repository accompanies the research paper, [*SimpleFold: Folding Proteins is Simpler than You Think*](https://arxiv.org/abs/2509.18480) (Arxiv 2025).

*Yuyang Wang, Jiarui Lu, Navdeep Jaitly, Joshua M. Susskind, Miguel Angel Bautista*

[[`Paper`](https://arxiv.org/abs/2509.18480)]  [[`BibTex`](#citation)]

<img src="assets/intro.png" width="750">

</div>


## Introduction

We introduce SimpleFold, the first flow-matching based protein folding model that solely uses general purpose transformer layers. SimpleFold does not rely on expensive modules like triangle attention or pair representation biases, and is trained via a generative flow-matching objective. We scale SimpleFold to 3B parameters and train it on more than 8.6M distilled protein structures together with experimental PDB data. To the best of our knowledge, SimpleFold is the largest scale folding model ever developed. On standard folding benchmarks, SimpleFold-3B model achieves competitive performance compared to state-of-the-art baselines. Due to its generative training objective, SimpleFold also demonstrates strong performance in ensemble prediction. SimpleFold challenges the reliance on complex domain-specific architectures designs in folding, highlighting an alternative yet important avenue of progress in protein structure prediction.

</div>


## Installation

To install `simplefold` package from github repository, run
```
git clone https://github.com/apple/ml-simplefold.git
cd ml-simplefold
conda create -n simplefold python=3.10
conda activate simplefold
python -m pip install -U pip build; pip install -e .
```
If you want to use MLX backend on Apple silicon: 
```
pip install mlx==0.28.0
pip install git+https://github.com/facebookresearch/esm.git
```

## Example 

We provide a jupyter notebook [`sample.ipynb`](sample.ipynb) to predict protein structures from example protein sequences. 

## Inference

Once you have `simplefold` package installed, you can predict the protein structure from target fasta file(s) via the following command line. We provide support for both [PyTorch](https://pytorch.org/) and [MLX](https://mlx-framework.org/) (recommended for Apple hardware) backends in inference. 
```
simplefold \
    --simplefold_model simplefold_100M \  # specify folding model in simplefold_100M/360M/700M/1.1B/1.6B/3B
    --num_steps 500 --tau 0.01 \        # specify inference setting
    --nsample_per_protein 1 \           # number of generated conformers per target
    --plddt \                           # output pLDDT
    --fasta_path [FASTA_PATH] \         # path to the target fasta directory or file
    --output_dir [OUTPUT_DIR] \         # path to the output directory
    --backend [mlx, torch]              # choose from MLX and PyTorch for inference backend 
```

## Evaluation

We provide predicted structures from SimpleFold of different model sizes:
```
https://ml-site.cdn-apple.com/models/simplefold/cameo22_predictions.zip # predicted structures of CAMEO22
https://ml-site.cdn-apple.com/models/simplefold/casp14_predictions.zip  # predicted structures of CASP14
https://ml-site.cdn-apple.com/models/simplefold/apo_predictions.zip     # predicted structures of Apo
https://ml-site.cdn-apple.com/models/simplefold/codnas_predictions.zip  # predicted structures of Fold-switch (CoDNaS)
```
We use the docker image of [openstructure](https://git.scicore.unibas.ch/schwede/openstructure/) 2.9.1 to evaluate generated structures for folding tasks (i.e., CASP14/CAMEO22). Once having the docker image enabled, you can run evaluation via:
```
python src/simplefold/evaluation/analyze_folding.py \
    --data_dir [PATH_TO_TARGET_MMCIF] \
    --sample_dir [PATH_TO_PREDICTED_MMCIF] \
    --out_dir [PATH_TO_OUTPUT] \
    --max-workers [NUMBER_OF_WORKERS]
```
To evaluate results of two-state prediction (i.e., Apo/CoDNaS), one need to compile the [TMsore](https://zhanggroup.org/TM-score/TMscore.cpp) and then run evaluation via:
```
python src/simplefold/evaluation/analyze_two_state.py \ 
    --data_dir [PATH_TO_TARGET_DATA_DIRECTORY] \
    --sample_dir [PATH_TO_PREDICTED_PDB] \
    --tm_bin [PATH_TO_TMscore_BINARY] \
    --task apo \ # choose from apo and codnas
    --nsample 5
```

## Train

You can also train or tune SimpleFold on your end. Instructions below include details for SimpleFold training. 

### Data preparation

#### Training targets

SimpleFold is trained on joint datasets including experimental structures from [PDB](https://www.rcsb.org/), as well as distilled predictions from [AFDB SwissProt](https://alphafold.ebi.ac.uk/download#swissprot-section) and [AFESM](https://afesm.foldseek.com/). Target lists of filtered SwissProt and AFESM targets thta are used in our training can be found:
```
https://ml-site.cdn-apple.com/models/simplefold/swissprot_list.csv # list of filted SwissProt (~270K targets)
https://ml-site.cdn-apple.com/models/simplefold/afesm_list.csv # list of filted AFESM targets (~1.9M targets)
https://ml-site.cdn-apple.com/models/simplefold/afesme_dict.json # list of filted extended AFESM (AFESM-E) (~8.6M targets)
```
In `afesme_dict.json`, the data is stored in the following structure:
```
{
    cluster 1 ID: {"members": [protein 1 ID, protein 2 ID, ...]},
    cluster 2 ID: {"members": [protein 1 ID, protein 2 ID, ...]},
    ...
}
```

Of course, one can use own customized datasets to train or tune SimpleFold models. Instructions below list how to process the dataset for SimpleFold training. 

#### Process mmcif structures

To process downloaded mmcif files, you need [Redis](https://redis.io/docs/latest/operate/oss_and_stack/install/archive/install-redis/) installed and launch the Redis server:
```
wget https://boltz1.s3.us-east-2.amazonaws.com/ccd.rdb
redis-server --dbfilename ccd.rdb --port 7777
```
You can then process mmcif files to input format for SimpleFold:
```
python src/simplefold/process_mmcif.py \
    --data_dir [MMCIF_DIR]   # directory of mmcif files
    --out_dir [OUTPUT_DIR]   # directory of processed targets
    --use-assembly
```
To further tokenize the processed structures:
```
python src/simplefold/process_structure.py \
    --target_dir [TARGET_DIR]   # directory of processed targets
    --token_dir [TOKEN_DIR]   # directory of tokenized data
```

### Training

The configuration of model is based on [`Hydra`](https://hydra.cc/docs/intro/). An example training configuration can be found in `configs/experiment/train`. To change dataset and model settings, one can refer to config files in `configs/data` and `configs/model`. To initiate SimpleFold training:
```
python train experiment=train
```
To train SimpleFold with FSDP strategy:
```
python train_fsdp.py experiment=train_fsdp
```

## Citation
If you found this code useful, please cite the following paper:
```
@article{simplefold,
  title={SimpleFold: Folding Proteins is Simpler than You Think},
  author={Wang, Yuyang and Lu, Jiarui and Jaitly, Navdeep and Susskind, Josh and Bautista, Miguel Angel},
  journal={arXiv preprint arXiv:2509.18480},
  year={2025}
}
```
SimpleFold was trained on experimental protein structures from the Protein Data Bank (PDB). The PDB is managed by the worldwide PDB (wwPDB) organization. Please cite the relevant PDB article(s) when using experimental structures:
```
@article{wwpdb2024,
  title={Protein Data Bank: The single global archive for 3D structural data of biological macromolecules},
  author={Berman, Helen and Henrick, Kim and Nakamura, Haruki},
  journal={Nucleic Acids Research},
  volume={51},
  number={D1},
  pages={D463--D471},
  year={2023},
  publisher={Oxford University Press}
}
```

## Acknowledgements
Our codebase is built using multiple opensource contributions, please see [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for more details. 

## License
Please check out the repository [LICENSE](LICENSE) before using the provided code and
[LICENSE_MODEL](LICENSE_MODEL) for the released models.
