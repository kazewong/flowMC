import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

#import jaxopt
from scipy.optimize import minimize
from utils import rv_model, get_kepler_params_and_log_jac, log_likelihood

jax.config.update("jax_enable_x64", True)


true_params = jnp.array([
    12.0, np.log(0.5), np.log(14.5), np.log(2.3), 
    np.sin(1.5), np.cos(1.5), 0.4, np.sin(-0.7), np.cos(-0.7)
])

random = np.random.default_rng(12345)
t = np.sort(random.uniform(0, 100, 50))
rv_err = 0.3
rv_obs = rv_model(true_params, t) + random.normal(0, rv_err, len(t))

plt.plot(t, rv_obs, ".k")
x = np.linspace(0, 100, 500)
plt.plot(x, rv_model(true_params, x), "C0")
plt.show(block=False)


# neg_logp_and_grad = jax.jit(lambda p: -log_likelihood(p, t, rv_err, rv_obs))
# solver = jaxopt.LBFGS(fun=neg_logp_and_grad, maxiter=1000)
# res = solver.run(true_params)

neg_logp_and_grad = jax.jit(jax.value_and_grad(lambda p: -log_likelihood(p, t, rv_err, rv_obs)))
soln = minimize(neg_logp_and_grad, true_params, jac=True)
pos_ini = jnp.asarray(get_kepler_params_and_log_jac(soln.x)[0])

pos_ini = pos_ini.repeat(10, 0)

jnp.asarray(get_kepler_params_and_log_jac(soln.x)[0])
