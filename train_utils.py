from functools import partial
import math

import jax
import jax.numpy as jnp
import flax
import optax
import diffrax
import numpy as np

from models import utils as mutils

def get_optimizer(config):
  schedule = optax.join_schedules([optax.linear_schedule(0.0, config.train.lr, config.train.warmup), 
                                   optax.constant_schedule(config.train.lr)], 
                                   boundaries=[config.train.warmup])
  optimizer = optax.adam(learning_rate=schedule, b1=config.train.beta1, eps=config.train.eps)
  optimizer = optax.chain(
    optax.clip(config.train.grad_clip),
    optimizer
  )
  return optimizer


def get_step_fn(optimizer, loss_fn):

  def step_fn(carry_state, batch):
    (rng, state) = carry_state
    rng, step_rng = jax.random.split(rng)
    grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
    (loss, new_sampler_state), grad = grad_fn(step_rng, state.model_params, state.sampler_state, batch)
    grad = jax.lax.pmean(grad, axis_name='batch')
    updates, opt_state = optimizer.update(grad, state.opt_state, state.model_params)
    new_params = optax.apply_updates(state.model_params, updates)
    new_params_ema = jax.tree_map(
      lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate),
      state.params_ema, new_params
    )
    new_state = state.replace(
      step=state.step+1,
      opt_state=opt_state,
      sampler_state=new_sampler_state, 
      model_params=new_params,
      params_ema=new_params_ema
    )

    loss = jax.lax.pmean(loss, axis_name='batch')
    new_carry_state = (rng, new_state)
    return new_carry_state, loss

  return step_fn


def get_artifact_generator(model, config, dynamics, artifact_shape):
  if 'am' == config.model.loss:
    generator = get_ode_generator(model, config, dynamics, artifact_shape)
  elif 'sam' == config.model.loss:  
    generator = get_sde_generator(model, config, dynamics, artifact_shape)
  elif 'ssm' == config.model.loss:  
    generator = get_ssm_generator(model, config, dynamics, artifact_shape)
  elif 'dsm' == config.model.loss:  
    generator = get_dsm_generator(model, config, dynamics, artifact_shape)
  else:
    raise NotImplementedError(f'generator for f{config.model.loss} is not implemented')
  return generator


def get_ode_generator(model, config, dynamics, artifact_shape):

  def artifact_generator(key, state, batch):
    x_0, _, _ = dynamics(key, batch, t=jnp.zeros((1)))
    s = mutils.get_model_fn(model, 
                            state.params_ema if config.eval.use_ema else state.model_params, 
                            train=False)
    def vector_field(t,y,args):
      dsdx = jax.grad(lambda _t, _x: s(_t*jnp.ones([x_0.shape[0],1,1,1]), _x).sum(), argnums=1)
      return dsdx(t,y)
    solve = partial(diffrax.diffeqsolve, 
                    terms=diffrax.ODETerm(vector_field), 
                    solver=diffrax.Euler(), 
                    t0=0.0, t1=1.0, dt0=1e-2, 
                    saveat=diffrax.SaveAt(ts=[1.0]),
                    stepsize_controller=diffrax.ConstantStepSize(True), 
                    adjoint=diffrax.NoAdjoint())
  
    solution = solve(y0=x_0)
    return solution.ys[-1][:,:,:,:artifact_shape[3]], solution.stats['num_steps']
    
  return artifact_generator


def get_sde_generator(model, config, dynamics, artifact_shape):

  def artifact_generator(key, state, batch):
    key, dynamics_key = jax.random.split(key)
    x_0, _, _ = dynamics(dynamics_key, batch, t=jnp.zeros((1)))

    s = mutils.get_model_fn(model, 
                            state.params_ema if config.eval.use_ema else state.model_params, 
                            train=False)
    def vector_field(t,y,args):
      dsdx = jax.grad(lambda _t, _x: s(_t*jnp.ones([x_0.shape[0],1,1,1]), _x).sum(), argnums=1)
      return dsdx(t,y)
    
    if config.model.anneal_sigma:
      diffusion = lambda t, y, args: config.model.sigma * jnp.ones(x_0.shape) * (1-t)
    else:
      diffusion = lambda t, y, args: config.model.sigma * jnp.ones(x_0.shape)
    brownian_motion = diffrax.UnsafeBrownianPath(shape=x_0.shape, key=key)
    terms = diffrax.MultiTerm(diffrax.ODETerm(vector_field), 
                              diffrax.WeaklyDiagonalControlTerm(diffusion, brownian_motion))
    solve = partial(diffrax.diffeqsolve, 
                    terms=terms, 
                    solver=diffrax.Euler(), 
                    t0=0.0, t1=1.0, dt0=1e-2, 
                    saveat=diffrax.SaveAt(ts=[1.0]),
                    stepsize_controller=diffrax.ConstantStepSize(True), 
                    adjoint=diffrax.NoAdjoint())

    solution = solve(y0=x_0)
    return solution.ys[-1][:,:,:,:artifact_shape[3]], solution.stats['num_steps']

  return artifact_generator


def get_ssm_generator(model, config, dynamics, artifact_shape):

  def artifact_generator(key, state, batch):
    key, gen_key = jax.random.split(key)
    x_0, _, _ = dynamics(gen_key, batch, t=jnp.zeros((1)))
    t_0 = jnp.zeros((x_0.shape[0],1,1,1))
    s = mutils.get_model_fn(model, 
                            state.params_ema if config.eval.use_ema else state.model_params, 
                            train=False)
    dt = 1e-2
    final_step_size = 6e-5
    def langevin_step(carry_state, step_key):
      prev_x, t = carry_state
      eps = jax.random.normal(step_key, shape=prev_x.shape)
      step_size = (final_step_size/dt**2)*(1 - t + dt)**2
      next_x = prev_x + 0.5*step_size*s(t, prev_x) + jnp.sqrt(step_size)*eps
      return (next_x, t), next_x

    def dxdt(carry_state, step_key):
      prev_x, t = carry_state
      langevin_keys = jax.random.split(step_key, 1000//int(1.0/dt))
      next_t = t + dt
      next_x = jax.lax.scan(langevin_step, (prev_x, next_t), langevin_keys)[0][0]
      return (next_x, next_t), next_x

    dxdt_keys = jax.random.split(key, int(1.0/dt)-1)
    x_1 = jax.lax.scan(dxdt, (x_0, t_0), dxdt_keys)[0][0]
    t_1 = jnp.ones((x_0.shape[0],1,1,1))
    langevin_keys = jax.random.split(key, 100)
    x_1 = jax.lax.scan(langevin_step, (x_1, t_1), langevin_keys)[0][0]
    num_steps = len(dxdt_keys)*10 + len(langevin_keys)
    return x_1[:,:,:,:artifact_shape[3]], num_steps

  return artifact_generator


def get_dsm_generator(model, config, dynamics, artifact_shape):

  beta_0 = 0.1
  beta_1 = 20.0
  beta = lambda t: (1-t)*beta_0 + t*beta_1
  sigma = lambda t: jnp.sqrt(-jnp.expm1(-t*beta_0-0.5*t**2*(beta_1-beta_0)))
  f = lambda t, x: -0.5*beta(t)*x
  g = lambda t, x: jnp.sqrt(beta(t))

  def artifact_generator(key, state, batch):
    x_0, _, _ = dynamics(key, batch, t=jnp.zeros((1)))

    s = mutils.get_model_fn(model, 
                            state.params_ema if config.eval.use_ema else state.model_params, 
                            train=False)
    
    def vector_field(t,y,args):
      score = -s(t*jnp.ones([x_0.shape[0],1,1,1]), y) / sigma(t)
      dxdt = f(t, y) - 0.5*g(t,y)**2*score
      return dxdt
    solve = partial(diffrax.diffeqsolve, 
                    terms=diffrax.ODETerm(vector_field), 
                    solver=diffrax.Euler(), 
                    t0=1.0, t1=1e-4, dt0=-1e-2, 
                    saveat=diffrax.SaveAt(ts=[1e-4]),
                    stepsize_controller=diffrax.ConstantStepSize(True), 
                    adjoint=diffrax.NoAdjoint())
  
    solution = solve(y0=x_0)
    return solution.ys[-1], solution.stats['num_steps']
    
  return artifact_generator


def stack_imgs(x, n=8, m=8):
    im_size = x.shape[2]
    big_img = np.zeros((n*im_size,m*im_size,x.shape[-1]),dtype=np.uint8)
    for i in range(n):
        for j in range(m):
            p = x[i*m+j] * 255
            p = p.clip(0, 255).astype(np.uint8)
            big_img[i*im_size:(i+1)*im_size, j*im_size:(j+1)*im_size, :] = p
    return big_img
