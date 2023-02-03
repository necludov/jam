# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions and modules related to model definition.
"""
from typing import Any

import flax
import optax
import functools
import jax.numpy as jnp
import jax
import numpy as np
# from utils import batch_mul


# The dataclass that stores all training states
@flax.struct.dataclass
class State:
  step: int
  opt_state: Any
  model_params: Any
  ema_rate: float
  params_ema: Any
  key: Any
  sampler_state: Any
  wandbid: Any


_MODELS = {}


def register_model(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _MODELS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_model(name):
  return _MODELS[name]


def init_model(rng, config):
  """ Initialize a `flax.linen.Module` model. """
  model_name = config.model.name
  model_def = functools.partial(get_model(model_name), config=config)
  cond_channels = 0
  try:
    cond_channels = config.data.cond_channels
  except AttributeError:
    pass
  x_shape = (jax.local_device_count(), config.data.image_size, config.data.image_size, config.data.num_channels + cond_channels)
  t_shape = (jax.local_device_count(), 1, 1, 1)
  fake_x = jnp.zeros(x_shape)
  fake_t = jnp.zeros(t_shape, dtype=jnp.int32)
  params_rng, dropout_rng = jax.random.split(rng)
  model = model_def()
  variables = model.init({'params': params_rng, 'dropout': dropout_rng}, fake_t, fake_x, train=True)
  # Variables is a `flax.FrozenDict`. It is immutable and respects functional programming
  init_model_state, initial_params = variables.pop('params')
  return model, init_model_state, initial_params


def get_model_fn(model, params, train=False):
  """Create a function to give the output of the score-based model.

  Args:
    model: A `flax.linen.Module` object the represent the architecture of score-based model.
    params: A dictionary that contains all trainable parameters.
    train: `True` for training and `False` for evaluation.

  Returns:
    A model function.
  """

  def model_fn(t, x, rng=None):
    """Compute the output of the score-based model.

    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.
      rng: If present, it is the random state for dropout

    Returns:
      A tuple of (model output, new mutable states)
    """
    variables = dict(params=params)
    if not train:
      return model.apply(variables, t, x, train=False, mutable=False)
    else:
      rngs = {'dropout': rng}
      return model.apply(variables, t, x, train=True, mutable=False, rngs=rngs)

  return model_fn


def to_flattened_numpy(x):
  """Flatten a JAX array `x` and convert it to numpy."""
  return np.asarray(x.reshape((-1,)))


def from_flattened_numpy(x, shape):
  """Form a JAX array with the given `shape` from a flattened numpy array `x`."""
  return jnp.asarray(x).reshape(shape)
