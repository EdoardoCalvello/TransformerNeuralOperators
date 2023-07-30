# Transformers for Lagrangian Data Assimilation
Build transformer-based architectures for performing data assimilation tasks

# To build up a conda environment from scratch, I did:
```bash
$ conda create --name transformers python=3.10
$ conda activate transformers
$ conda install pytorch torchvision -c pytorch
$ conda install matplotlib
$ conda install -c conda-forge pytorch-lightning 
$ conda install -c conda-forge wandb 
$ conda install -c conda-forge torchdiffeq
```

# Then I exported this conda environment for reproducibility:
```bash
$ conda env export > transformers.yml
```

# If you are using this repo, please first try to install dependencies by simply running:
```bash
$ conda env create -f transformers.yml
```
- Note: This did not work for setting up on caltech HPC, so I used the above conda commands to build up from scratch.

# Set up a wandb account by following instructions here:
[wandb](https://wandb.ai/site)

# To perform preliminary training run of transformer to learns a mapping from a trajectory of L63's first coordinate to its third coordinate, run:
```bash
$ conda activate transformers
$ wandb login
$ python l63assimTRANS.py
```
