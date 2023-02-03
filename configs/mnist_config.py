import ml_collections


def get_config():
  config = ml_collections.ConfigDict()

  config.seed = 0

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'MNIST'
  data.ndims = 3
  data.image_size = 32
  data.num_channels = 1
  data.uniform_dequantization = True
  data.norm_mean = (0.5)
  data.norm_std = (0.5)
  data.random_flip = False
  data.task = 'generate'
  data.dynamics = 'generation'
  data.t_0, data.t_1 = 0.0, 1.0

  # model
  config.model = model = ml_collections.ConfigDict()
  model.name = 'anet'
  model.loss = 'sam'
  model.sigma = 1e-1
  model.anneal_sigma = False
  model.ema_rate = 0.99
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 32
  model.ch_mult = (1, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.dropout = 0.1
  
  # training
  config.train = train = ml_collections.ConfigDict()
  train.batch_size = 128
  train.n_jitted_steps = 1
  train.n_iters = 200_000
  train.save_every = 5_000
  train.eval_every = 5_000
  train.log_every = 50
  train.lr = 2e-4
  train.beta1 = 0.9
  train.eps = 1e-8
  train.warmup = 5_000
  train.grad_clip = 1.  

  # evaluation
  config.eval = eval = ml_collections.ConfigDict()
  eval.batch_size = 128
  eval.artifact_size = 64
  eval.num_samples = 500
  eval.use_ema = False
  eval.estimate_bpd = False

  return config
