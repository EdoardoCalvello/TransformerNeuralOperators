# Transformer Neural Operators
Build transformer-based architectures for performing data assimilation tasks

# To build up a conda environment, I did:
```bash
$ conda create --name transformers python=3.10
$ conda activate transformers
# (from [pytorch website](https://pytorch.org/get-started/locally/)
# if on HPC with cuda 11.8
    $ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# if on Mac with CPU only
    $ conda install pytorch torchvision -c pytorch
# Then proceed with
$ conda install matplotlib
$ conda install -c conda-forge pytorch-lightning 
$ conda install -c conda-forge wandb 
$ conda install -c conda-forge torchdiffeq
$ conda install -c conda-forge scikit-learn
```

# Then I exported this conda environment for reproducibility:
```bash
# if on HPC with cuda 11.8
$ conda env export > transformers_cuda11.8.yml

# if on Mac with CPU only
$ conda env export > transformers.yml
```

# If you are using this repo, please first try to install dependencies by simply running:
```bash
# if on HPC with cuda 11.8
$ conda env create -f transformers_cuda11.8.yml

# if on Mac with CPU only
$ conda env create -f transformers.yml
```
- Note: If this does not work, try the above instructions for installing from scratch via anaconda.

# Set up a wandb account by following instructions here:
[wandb](https://wandb.ai/site)

# To perform preliminary training run of transformer to learns a mapping from a trajectory of L63's first coordinate to its third coordinate, run:
```bash
$ conda activate transformers
$ wandb login
$ python l63assimTRANS.py
```
