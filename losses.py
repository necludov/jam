import math

import flax
import jax
import jax.numpy as jnp
import jax.random as random
from models import utils as mutils
from dynamics import dynamics
import dynamics.utils as dutils


def get_loss(config, model, q_t, time_sampler, train):
  if 'am' == config.model.loss:
    loss_fn = get_am_loss(config, model, q_t, time_sampler, train)
  elif 'sam' == config.model.loss:
    loss_fn = get_stoch_am_loss(config, model, q_t, time_sampler, train)
  elif 'ssm' == config.model.loss:
    loss_fn = get_ssm_loss(config, model, q_t, time_sampler, train)
  elif 'dsm' == config.model.loss:
    loss_fn = get_dsm_loss(config, model, q_t, time_sampler, train)
  else:
    raise NotImplementedError(f'loss {config.model.loss} is not implemented')
  return loss_fn


def get_am_loss(config, model, q_t, time_sampler, train):

  w_t_fn = lambda t: (1-t)
  dwdt_fn = jax.grad(lambda t: w_t_fn(t).sum(), argnums=0)

  def am_loss(key, params, sampler_state, batch):
    keys = random.split(key, num=5)
    s = mutils.get_model_fn(model, params, train=train)
    dsdtdx_fn = jax.grad(lambda t,x,_key: s(t,x,_key).sum(), argnums=[0,1])
    
    data = batch['image']
    bs = data.shape[0]

    # sample time
    t_0, t_1 = config.data.t_0*jnp.ones((bs,1,1,1)), config.data.t_1*jnp.ones((bs,1,1,1))
    t, next_sampler_state = time_sampler.sample_t(bs, sampler_state)
    t = jnp.expand_dims(t, (1,2,3))
    # sample data
    x_0, x_1, x_t = q_t(keys[0], data, t, t_0, t_1)

    # boundaries loss
    s_0 = s(t_0, x_0, rng=keys[1])
    s_1 = s(t_1, x_1, rng=keys[2])
    loss = w_t_fn(t_0)*s_0.reshape((-1,1,1,1)) - w_t_fn(t_1)*s_1.reshape((-1,1,1,1))
    print(loss.shape, 'boundaries.shape')

    # time loss
    dsdt, dsdx = dsdtdx_fn(t, x_t, keys[3])
    p_t = time_sampler.invdensity(t)
    s_t = s(t, x_t, keys[4])
    print(p_t.shape, dsdt.shape, dsdx.shape, 'p_t.shape, dsdt.shape, dsdx.shape')
    loss += w_t_fn(t)*p_t*(dsdt + 0.5*(dsdx**2).sum((1,2,3), keepdims=True))
    loss += s_t.reshape((-1,1,1,1))*dwdt_fn(t)*p_t
    print(loss.shape, 'final.shape')
    return loss.mean(), next_sampler_state

  return am_loss


def get_stoch_am_loss(config, model, q_t, time_sampler, train):

  w_t_fn = lambda t: (1-t)
  dwdt_fn = jax.grad(lambda t: w_t_fn(t).sum(), argnums=0)

  if config.model.anneal_sigma:
    sigma = lambda t: config.model.sigma * (1-t)
  else:
    sigma = lambda t: config.model.sigma

  def sam_loss(key, params, sampler_state, batch):
    keys = random.split(key, num=7)
    s = mutils.get_model_fn(model, params, train=train)
    dsdtdx_fn = jax.grad(lambda _t,_x,_key: s(_t,_x,_key).sum(), argnums=[0,1])
    
    data = batch['image']
    bs = data.shape[0]

    # sample time
    t_0, t_1 = config.data.t_0*jnp.ones((bs,1,1,1)), config.data.t_1*jnp.ones((bs,1,1,1))
    t, next_sampler_state = time_sampler.sample_t(bs, sampler_state)
    t = jnp.expand_dims(t, (1,2,3))
    # sample data
    x_0, x_1, x_t = q_t(keys[0], data, t)

    # boundaries loss
    s_0 = s(t_0, x_0, rng=keys[1])
    s_1 = s(t_1, x_1, rng=keys[2])
    loss = w_t_fn(t_0)*s_0.reshape((-1,1,1,1)) - w_t_fn(t_1)*s_1.reshape((-1,1,1,1))
    print(loss.shape, 'boundaries.shape')

    # time loss
    eps = random.randint(keys[3], x_t.shape, 0, 2).astype(float)*2 - 1.0
    dsdx_val, jvp_val, dsdt_val = jax.jvp(lambda _x: dsdtdx_fn(t, _x, keys[4])[::-1], (x_t,), (eps,), has_aux=True)
    s_t = s(t, x_t, keys[5])
    p_t = time_sampler.invdensity(t)
    print(p_t.shape, dsdt_val.shape, dsdx_val.shape, 'p_t.shape, dsdt.shape, dsdx.shape')
    time_loss = dsdt_val + 0.5*(dsdx_val**2).sum((1,2,3), keepdims=True)
    time_loss += 0.5*sigma(t)**2*(jvp_val*eps).sum((1,2,3), keepdims=True)
    time_loss *= w_t_fn(t)
    time_loss += s_t.reshape((-1,1,1,1))*dwdt_fn(t)
    loss += p_t*time_loss
    print(loss.shape, 'final.shape')
    return loss.mean(), next_sampler_state

  return sam_loss


def get_ssm_loss(config, model, q_t, time_sampler, train):

  w_t_fn = lambda t: (1-t)**2

  def ssm_loss(key, params, sampler_state, batch):
    keys = random.split(key, num=7)
    s = mutils.get_model_fn(model, params, train=train)
    
    data = batch['image']
    bs = data.shape[0]

    # sample time
    t, next_sampler_state = time_sampler.sample_t(bs, sampler_state)
    t = jnp.expand_dims(t, (1,2,3))
    # sample data
    _, _, x_t = q_t(keys[0], data, t)

    eps = random.randint(keys[3], x_t.shape, 0, 2).astype(float)*2 - 1.0
    s_val, jvp_val = jax.jvp(lambda _x: s(t, _x, keys[4]), (x_t,), (eps,))
    p_t = time_sampler.invdensity(t)
    print(p_t.shape, s_val.shape, jvp_val.shape, 'p_t.shape, s_val.shape, jvp_val.shape')
    loss = (jvp_val*eps).sum((1,2,3), keepdims=True) + 0.5*(s_val**2).sum((1,2,3), keepdims=True)
    loss *= w_t_fn(t)*p_t
    print(loss.shape, 'final.shape')
    return loss.mean(), next_sampler_state

  return ssm_loss


def get_dsm_loss(config, model, q_t, time_sampler, train):

  def dsm_loss(key, params, sampler_state, batch):
    keys = random.split(key, num=7)
    s = mutils.get_model_fn(model, params, train=train)
    
    data = batch['image']
    bs = data.shape[0]

    # sample time
    t, next_sampler_state = time_sampler.sample_t(bs, sampler_state)
    t = jnp.expand_dims(t, (1,2,3))
    # sample data
    eps, _, x_t = q_t(keys[0], data, t)

    # eval loss
    loss = ((eps - s(t, x_t, keys[1])) ** 2).sum((1,2,3))
    print(loss.shape, 'final.shape')
    return loss.mean(), next_sampler_state

  return dsm_loss
