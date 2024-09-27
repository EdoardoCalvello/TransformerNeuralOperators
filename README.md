## Transformer Neural Operators
Transformer Neural Operator architechtures and experiments from the paper "Continuum Attention for Neural Operators" (https://arxiv.org/abs/2406.06486)

```
@article{Calvello2024Continuum,
  title={Continuum Attention for Neural Operators},
  author={Calvello, Edoardo and Kovachki, Nikola B and Levine, Matthew E and Stuart, Andrew M},
  journal={arXiv preprint arXiv:2406.06486},
  year={2024}
}
```

## Setting up conda environment:
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
```bash
# if on HPC with cuda 11.8
$ conda env export > transformers_cuda11.8.yml

# if on Mac with CPU only
$ conda env export > transformers.yml
```

If you are using this repo, please first try to install dependencies by simply running:
```bash
# if on HPC with cuda 11.8
$ conda env create -f transformers_cuda11.8.yml

# if on Mac with CPU only
$ conda env create -f transformers.yml
```

