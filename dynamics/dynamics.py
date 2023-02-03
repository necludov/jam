import jax
import jax.random as random
import jax.numpy as jnp

from . import utils

@utils.register_dynamics(name='generation')
def generation(key, data, t, t0=0.0, t1=1.0):
  noise = random.normal(key, shape=data.shape)
  get_x_t = lambda t: (1-t)*noise + t*data
  return get_x_t(t0), get_x_t(t1), get_x_t(t)

@utils.register_dynamics(name='superres')
def superres(key, data, t):
  x_0 = random.normal(key, shape=data.shape)
  x_1 = data
  downscaled_shape = (data.shape[0], data.shape[1]//2, data.shape[2]//2, data.shape[3])
  downscaled_x = jax.image.resize(x_1, downscaled_shape, method='nearest')
  downscaled_x = jax.image.resize(downscaled_x, x_1.shape, method='bilinear')
  x_0 = x_0.at[:,1::2,:,:].set(downscaled_x[:,1::2,:,:])
  x_0 = x_0.at[:,:,1::2,:].set(downscaled_x[:,:,1::2,:])
  x_0 = jax.lax.concatenate([x_0, downscaled_x], 3)
  x_1 = jax.lax.concatenate([x_1, downscaled_x], 3)
  x_t = (1-t)*x_0 + t*x_1
  return x_0, x_1, x_t

@utils.register_dynamics(name='color')
def color(key, data, t):
  grayscale = data.mean(-1, keepdims=True)
  grayscale = jnp.tile(grayscale, (1,1,1,3))
  x_1 = data
  x_0 = (1e-1*random.normal(key, data.shape) + grayscale)
  x_0 = jax.lax.concatenate([x_0, grayscale], 3)
  x_1 = jax.lax.concatenate([x_1, grayscale], 3)
  x_t = t*x_1 + (1-t)*x_0
  return x_0, x_1, x_t

@utils.register_dynamics(name='vpsde')
def vpsde(key, data, t):
  beta_0 = 0.1
  beta_1 = 20.0
  beta = lambda t: (1-t)*beta_0 + t*beta_1
  alpha = lambda t: jnp.exp(-0.5*t*beta_0-0.25*t**2*(beta_1-beta_0))
  sigma = lambda t: jnp.sqrt(-jnp.expm1(-t*beta_0-0.5*t**2*(beta_1-beta_0)))

  x_0 = random.normal(key, shape=data.shape)
  x_1 = data
  x_t = sigma(t)*x_0 + alpha(t)*x_1
  return x_0, x_1, x_t
