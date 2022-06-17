import jax
import jax.numpy as jnp                # JAX NumPy
from jax.scipy.special import logsumexp
import numpy as np  
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import corner
import tqdm
import time
import pickle

from flowMC.nfmodel.realNVP import RealNVP
from flowMC.sampler.MALA import mala_sampler
from flowMC.nfmodel.utils import *

from utils import rv_model, log_likelihood, log_prior, sample_prior, get_kepler_params_and_log_jac
from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys

jax.config.update("jax_enable_x64", True)

## Generate probelm
true_params = jnp.array([
    12.0, # v0
    np.log(0.5), # log_s2
    np.log(14.5), # log_P
    np.log(2.3), # log_k
    np.sin(1.5), # phi
    np.cos(1.5),
    0.4, # ecc
    np.sin(-0.7), # w
    np.cos(-0.7)
])

prior_kwargs = {
    'ecc_alpha': 2, 'ecc_beta': 2,
    'log_k_mean': 1, 'log_k_var': 1,
    'v0_mean': 10, 'v0_var': 2,
    'log_period_mean': 2.5, 'log_period_var': 0.5,
    'log_s2_mean': -0.5, 'log_s2_var': 0.1,
}

random = np.random.default_rng(12345)
t = np.sort(random.uniform(0, 100, 50))
rv_err = 0.3
sigma2 = rv_err ** 2 + jnp.exp(2 * true_params[1])
rv_obs = rv_model(true_params, t) + random.normal(0, sigma2, len(t))
# plt.plot(t, rv_obs, ".k")
# x = np.linspace(0, 100, 500)
# plt.plot(x, rv_model(true_params, x), "C0")
# plt.show(block=False)

## Setting up sampling -- takes one input at the time.
def log_posterior(x):
    return  log_likelihood(x.T, t, rv_err, rv_obs) + log_prior(x,**prior_kwargs)
    
d_log_posterior = jax.grad(log_posterior)

config = {}
n_dim = 9
n_loop = 2
n_local_steps = 5
n_global_steps = 2
n_chains = 5
learning_rate = 0.01
momentum = 0.9
num_epochs = 5
batch_size = 10
stepsize = 1e-3
logging = True

print("Preparing RNG keys")
rng_key_set = initialize_rng_keys(n_chains,seed=42)

print("Initializing MCMC model and normalizing flow model.")

# initial_position = jax.random.normal(rng_key_set[0],shape=(n_chains,n_dim)) #(n_chains, n_dim)

kepler_params_ini = sample_prior(rng_key_set[0], n_chains,
                                 **prior_kwargs)
neg_logp_and_grad = jax.jit(jax.value_and_grad(lambda p: -log_posterior(p)))
optimized = []
for i in tqdm.tqdm(range(n_chains)):
    soln = minimize(neg_logp_and_grad, kepler_params_ini.T[i].T, jac=True)
    optimized.append(jnp.asarray(get_kepler_params_and_log_jac(soln.x)[0]))

initial_position = jnp.stack(optimized) #(n_chains, n_dim)
# initial_position = kepler_params_ini.T

mean = initial_position.mean(0)
init_centered = (initial_position - mean)
cov = init_centered.T @ init_centered / n_chains

model = RealNVP(10, n_dim, 64, 1)

run_mala = jax.vmap(mala_sampler, in_axes=(0, None, None, None, 0, None),
                    out_axes=0)

print("Initializing sampler class")

nf_sampler = Sampler(n_dim, rng_key_set, model, run_mala,
                    log_posterior,
                    d_likelihood=d_log_posterior,
                    n_loop=n_loop,
                    n_local_steps=n_local_steps,
                    n_global_steps=n_global_steps,
                    n_chains=n_chains,
                    n_epochs=num_epochs,
                    n_nf_samples=100,
                    learning_rate=learning_rate,
                    momentum=momentum,
                    batch_size=batch_size,
                    stepsize=stepsize)


print("Sampling")

start = time.time()
_ = nf_sampler.sample(initial_position)
chains, nf_samples, local_accs, global_accs, loss_vals = _

print('Elapsed: ', time.time()-start, 's')

print("Make plots")

chains = np.array(chains)
nf_samples = np.array(nf_samples)

value1 = true_params
n_dim = n_dim
# Make the base corner plot

figure = corner.corner(chains.reshape(-1,n_dim),
                labels=['v0', 'log_s2', 'log_period', 'log_k', 'sin_phi_', 'cos_phi_', 'ecc_', 'sin_w_', 'cos_w_'], title_kwargs={"fontsize": 12})
figure.set_size_inches(7, 7)
figure.suptitle('Visualize chains')
# Extract the axes
axes = np.array(figure.axes).reshape((n_dim, n_dim))
# Loop over the diagonal
for i in range(n_dim):
    ax = axes[i, i]
    ax.axvline(value1[i], color="g")
# Loop over the histograms
for yi in range(n_dim):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.plot(value1[xi], value1[yi], "sg")
        ax.plot(chains[1, -1000:, xi],chains[1, -1000:, yi], alpha=0.75, lw=0.75)
        ax.plot(chains[0, -1000:, xi],chains[0, -1000:, yi], alpha=0.75, lw=0.75)
        ax.axvline(value1[xi], color="g")
        ax.axhline(value1[yi], color="g")
# plt.tight_layout()
plt.show(block=False)

plt.figure(figsize=(10,5))
axs = [plt.subplot(121), plt.subplot(122)]
plt.sca(axs[0])
plt.plot(t, rv_obs, ".k", label='observations')
x = np.linspace(0, 100, 500)
plt.plot(x, rv_model(true_params, x), "C0", label='ground truth')
for i in range(np.minimum(n_chains,10)):
    params, log_jac = get_kepler_params_and_log_jac(chains[i,-1,:])
    if i == 0:
        plt.plot(x, rv_model(params, x), c='gray', alpha=0.5, label='final samples')
    else:
        plt.plot(x, rv_model(params, x), c='gray', alpha=0.5)
plt.xlabel('t')
plt.ylabel('radial velocity')
plt.legend()

plt.sca(axs[1])
posterior_evolution = jax.vmap(log_posterior)(chains[-10:,:,:].reshape(-1, 9)).reshape(10, -1)
shift = max(jnp.max(posterior_evolution),0)
plt.plot(- (posterior_evolution.T - shift))
plt.yscale('log')
plt.ylabel('walker negative log-likelihood')
plt.xlabel('iteration')

plt.show(block=False)

value1 = true_params
n_dim = n_dim
# Make the base corner plot
figure = corner.corner(chains.reshape(-1,n_dim),
                labels=['v0', 'log_s2', 'log_period', 'log_k', 'sin_phi_', 'cos_phi_', 'ecc_', 'sin_w_', 'cos_w_'], title_kwargs={"fontsize": 12})
figure.set_size_inches(7, 7)
figure.suptitle('Visualize initializations')
# Extract the axes
axes = np.array(figure.axes).reshape((n_dim, n_dim))
# Loop over the diagonal
for i in range(n_dim):
    ax = axes[i, i]
    ax.axvline(value1[i], color="g")
# Loop over the histograms
for yi in range(n_dim):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.plot(value1[xi], value1[yi], "sg")
        ax.plot(initial_position[:,xi], initial_position[:,yi],'+',ms=2)
        ax.axvline(value1[xi], color="g")
        ax.axhline(value1[yi], color="g")
# plt.tight_layout()
plt.show(block=False)


results = {
    'chains': chains,
    'nf_samples': nf_samples,
    'prior_samples': kepler_params_ini,
    'optimized_init': initial_position,
    'config': config,
    'true_params': true_params,
    'rv_obs': rv_obs,
    't': t,
    'prior_kwargs': prior_kwargs,
    'rv_err': rv_err
}

random_id = np.random.randint(10000)
with open('results_{:d}.pkl'.format(random_id), 'wb') as f:
    pickle.dump(results, f)