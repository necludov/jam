from typing import Any
from functools import partial

import math

import jax
import jax.numpy as jnp
import flax
import diffrax

from models import utils as mutils


@flax.struct.dataclass
class EvalState:
  bpd_batch_id: int
  sample_batch_id: int
  global_iter: int
  key: Any


def get_bpd_estimator(model, config):
  if 'am' == config.model.loss:
    get_bpd = get_am_bpd_estimator(model, config)
  elif 'dsm' == config.model.loss:  
    get_bpd = get_dsm_bpd_estimator(model, config)
  else:
    raise NotImplementedError(f'bpd estimator for f{config.model.loss} is not implemented')
  return get_bpd


def get_am_bpd_estimator(model, config):

  def get_bpd(key, state, batch):
    x_1 = batch['image']
    key, eps_key = jax.random.split(key)
    eps = jax.random.randint(eps_key, x_1.shape, 0, 2).astype(float)*2 - 1.0

    def vector_field(t,data,args):
      state = args[0]
      x, log_p = data
      s = mutils.get_model_fn(model, 
                              state.params_ema if config.eval.use_ema else state.model_params, 
                              train=False)
      dsdx = jax.grad(lambda _t, _x: s(_t, _x).sum(), argnums=1)
      dsdx_val, jvp_val = jax.jvp(lambda _x: dsdx(t*jnp.ones([x_1.shape[0],1,1,1]), _x), (x,), (eps,))
      return (dsdx_val, (jvp_val*eps).sum((1,2,3))) # mind that dt is negative in the solver

    t0, t1 = 0.0, 1.0
    solve = partial(diffrax.diffeqsolve, 
                    terms=diffrax.ODETerm(vector_field), 
                    solver=diffrax.Dopri5(), 
                    t0=t1, t1=t0, dt0=-1e-4, 
                    saveat=diffrax.SaveAt(ts=[t0]),
                    stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5), 
                    adjoint=diffrax.NoAdjoint())
  
    solution = solve(y0=(x_1, jnp.zeros(x_1.shape[0])), args=(state,))
    x_0, delta_log_p = solution.ys[0][-1], solution.ys[1][-1]
    D = jnp.array(x_0.shape[1:]).prod()
    log_p_0 = -0.5*(x_0**2).sum((1,2,3)) - 0.5*D*math.log(2*math.pi)
    log_p_1 = log_p_0 + delta_log_p
    bpd = -log_p_1 / math.log(2) / D + 7.0
    return jax.lax.pmean(bpd.mean(), axis_name='batch'), solution.stats['num_steps']

  return get_bpd


def get_dsm_bpd_estimator(model, config):

  beta_0 = 0.1
  beta_1 = 20.0
  beta = lambda t: (1-t)*beta_0 + t*beta_1
  sigma = lambda t: jnp.sqrt(-jnp.expm1(-t*beta_0-0.5*t**2*(beta_1-beta_0)))
  f = lambda t, x: -0.5*beta(t)*x
  g = lambda t, x: jnp.sqrt(beta(t))

  def get_bpd(key, state, batch):
    x_0 = batch['image']
    key, eps_key = jax.random.split(key)
    eps = jax.random.randint(eps_key, x_0.shape, 0, 2).astype(float)*2 - 1.0

    def vector_field(t,data,args):
      state = args[0]
      x, log_p = data
      s = mutils.get_model_fn(model, 
                              state.params_ema if config.eval.use_ema else state.model_params, 
                              train=False)
      dxdt = lambda _x: f(t, _x) - 0.5*g(t,_x)**2*(-s(t*jnp.ones([x_0.shape[0],1,1,1]), _x) / sigma(t))
      dxdt_val, jvp_val = jax.jvp(dxdt, (x,), (eps,))
      return (dxdt_val, (jvp_val*eps).sum((1,2,3)))

    solve = partial(diffrax.diffeqsolve, 
                    terms=diffrax.ODETerm(vector_field), 
                    solver=diffrax.Dopri5(), 
                    t0=1e-4, t1=1.0, dt0=1e-4, 
                    saveat=diffrax.SaveAt(ts=[1.0]),
                    stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5),
                    adjoint=diffrax.NoAdjoint())
    
    solution = solve(y0=(x_0, jnp.zeros(x_0.shape[0])), args=(state, ))
    x_1, delta_log_p = solution.ys[0][-1], solution.ys[1][-1]
    D = jnp.array(x_1.shape[1:]).prod()
    log_p_1 = -0.5*(x_1**2).sum((1,2,3)) - 0.5*D*math.log(2*math.pi)
    log_p_0 = log_p_1 + delta_log_p
    bpd = -log_p_0 / math.log(2) / D + 7.0
    return jax.lax.pmean(bpd.mean(), axis_name='batch'), solution.stats['num_steps']

  return get_bpd


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
    t0, t1 = 0.0, 1.0
    solve = partial(diffrax.diffeqsolve, 
                    terms=diffrax.ODETerm(vector_field), 
                    solver=diffrax.Dopri5(), 
                    t0=t0, t1=t1, dt0=1e-4, 
                    saveat=diffrax.SaveAt(ts=[t1]),
                    stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5), 
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

    diffusion = lambda t, y, args: config.model.sigma * jnp.ones(x_0.shape) * (1-t)
    brownian_motion = diffrax.VirtualBrownianTree(t0=0.0, t1=1.0, tol=1e-5, shape=x_0.shape, key=key)
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
                    solver=diffrax.Dopri5(), 
                    t0=1.0, t1=1e-4, dt0=-1e-4, 
                    saveat=diffrax.SaveAt(ts=[1e-4]),
                    stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5), 
                    adjoint=diffrax.NoAdjoint())
  
    solution = solve(y0=x_0)
    return solution.ys[-1], solution.stats['num_steps']
    
  return artifact_generator
