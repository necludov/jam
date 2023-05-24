# jAM (JAX implementation of [Action Matching](https://arxiv.org/abs/2210.06662))

Action Matching is a method for learning the time evolution of distributions from samples. That is, suppose we observe the time evolution of some random variable $X_t$ with the density $q_t$, from $t_0$ to $t_1$. 
Having access to uncorrelated samples of $X_t$ (which are not a part of a single trajectory) at different points in time $t\in [t_0,t_1]$, 
Action Matching learns a model of the dynamics by learning how to move samples in time such that they are distributed according to $q_t$ for any $t$.

The key idea is to learn such function $s_t$ (which we parameterize with a neural network) that yields the continuity equation
$$\partial_t q_t = -\nabla \cdot(q_t \nabla s_t).$$
Then the propagation of samples in time can be performed simply by simulating the corresponding ODE
$$\frac{dx}{dt} = \nabla s_t(x).$$
For the method description see the [Tutorials section](https://github.com/necludov/jam#tutorials) below or [the paper](https://arxiv.org/abs/2210.06662).


## Tutorials
|Method|Link|
|:----|:-----:|
|Action Matching|[![Open In Colab](https://github.com/necludov/jam/blob/main/colab-badge.svg)](https://colab.research.google.com/drive/1-vGU7r8rvsA2m0VWQvzfnsn2pUWfOuYL?usp=sharing)|
|Entropic (Stochastic) Action Matching|[![Open In Colab](https://github.com/necludov/jam/blob/main/colab-badge.svg)](https://colab.research.google.com/drive/1e25hnB0jVr-bTnzEMQgXuVMsZqYfuTlH?usp=sharing)|
|Unbalanced Action Matching (with reweighting)|[![Open In Colab](https://github.com/necludov/jam/blob/main/colab-badge.svg)](https://colab.research.google.com/drive/1jnT1A8HI8RGIuLCokdvnm6NZ2f6wONdR?usp=sharing)|

## Running the code

Run all the code using `main.py` with different `config` and `mode`.
* `config` flag takes the path to the config file.
* `mode` flag takes one of the following values: "train", "eval", "fid_stats". All the modes require a config file. Mind the `data.uniform_dequantization` flag when evaluating statistics on the dataset for FID evaluation.
* `workdir` is the path to the working directory for storing states.

For instance, on the clusters with slurm, you would run the code like this inside your sbatch scripts
```
python main.py --config configs/am/cifar/generation.py \
               --workdir $PWD/checkpoint/${SLURM_JOB_ID} \
               --mode 'train'
```

