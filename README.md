# Learning Multi-Level Representations for Hierarchical Music Structure Analysis
This repository contains a [PyTorch](http://pytorch.org/) implementation of the paper [Learning Multi-Level Representations for Hierarchical Music Structure Analysis](https://hal.archives-ouvertes.fr/hal-03780032/) 
presented at ISMIR 2022.

The code is based on the implementation of [Conditional Similarity Networks](https://arxiv.org/abs/1603.07810) and the overall format used in the
[MSAF](https://ismir2015.ismir.net/LBD/LBD30.pdf) package. 

## Table of Contents
0. [Usage](#usage)
0. [Requirements] (#requirements)
0. [Citing](#citing)
0. [Contact](#contact)

## Usage
The detault setting for this repo is a CSN with fixed masks, an embedding dimension 128 and four notions of temporal distance (from the coarsest
to the most refined). The baseline denoted as [Flat embeddings](https://ieeexplore.ieee.org/document/8683407) can be 
obtained by setting the n_conditions parameter to 1.

The network can be trained with:

```
python exp.py --feat_id {feature type} --ds_path {path to the dataset}
```

The dataset format should follow:
```
dataset/
├── audio                   # audio files (.mp3, .wav, .aiff)
├── features                # feature files (.npy)
└── references              # references files (.jams)
```

## Requirements
```
conda env create -f environment.yml
```

## Citing
```
@inproceedings{buisson2022learning,
  title={Learning Multi-Level Representations for Hierarchical Music Structure Analysis},
  author={Buisson, Morgan and Mcfee, Brian and Essid, Slim and Crayencour, H{\'e}l{\`e}ne C},
  booktitle={International Society for Music Information Retrieval (ISMIR)},
  year={2022}
}
```

## Contact
morgan.buisson@telecom-paris.fr
