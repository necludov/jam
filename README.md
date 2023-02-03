# jAM (JAX implementation of [Action Matching](https://arxiv.org/abs/2210.06662))

## Tutorials
|Method|Link|
|:----|:-----:|
|Action Matching|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-vGU7r8rvsA2m0VWQvzfnsn2pUWfOuYL?usp=sharing)|
|Entropic (Stochastic) Action Matching|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jnT1A8HI8RGIuLCokdvnm6NZ2f6wONdR?usp=sharing)|
|Unbalanced Action Matching (with reweighting)|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1e25hnB0jVr-bTnzEMQgXuVMsZqYfuTlH?usp=sharing)|

## Running the code

Run all the code using `main.py` with different `config` and `mode`.
* `config` flag takes the path to the config file.
* `mode` flag takes one of the following values: "train", "eval", "fid_stats". All the modes require a config file. Mind the `data.uniform_dequantization` mode when evaluating statistics on the dataset for FID evaluation.
* `workdir` is the path to the working directory for storing states.

For instance, on the clusters with slurm, you would run the code like this inside your sbatch scripts
```
python main.py --config configs/am/cifar/generation.py \
               --workdir $PWD/checkpoint/${SLURM_JOB_ID} \
               --mode 'train'
```

