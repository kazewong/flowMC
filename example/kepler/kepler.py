import jax
import jax.numpy as jnp                # JAX NumPy
from jax.scipy.special import logsumexp
import numpy as np  
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import corner
import tqdm
import time

from nfsampler.nfmodel.realNVP import RealNVP
from nfsampler.sampler.MALA import mala_sampler
from nfsampler.nfmodel.utils import *

from utils import rv_model, log_likelihood, log_prior, sample_prior, get_kepler_params_and_log_jac
from nfsampler.utils import Sampler, initialize_rng_keys

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
config['n_dim'] = 9
config['n_loop'] = 10
config['n_samples'] = 100
config['nf_samples'] = 10
config['n_chains'] = 100
config['learning_rate'] = 0.01
config['momentum'] = 0.9
config['num_epochs'] = 5
config['batch_size'] = 100
config['stepsize'] = 1e-2
config['logging'] = True

print("Preparing RNG keys")
rng_key_set = initialize_rng_keys(config['n_chains'],seed=42)

print("Initializing MCMC model and normalizing flow model.")

# initial_position = jax.random.normal(rng_key_set[0],shape=(config['n_chains'],config['n_dim'])) #(n_chains, n_dim)

kepler_params_ini = sample_prior(rng_key_set[0], config['n_chains'],
                                 **prior_kwargs)
neg_logp_and_grad = jax.jit(jax.value_and_grad(lambda p: -log_posterior(p)))
optimized = []
for i in tqdm.tqdm(range(config['n_chains'])):
    soln = minimize(neg_logp_and_grad, kepler_params_ini.T[i].T, jac=True)
    optimized.append(jnp.asarray(get_kepler_params_and_log_jac(soln.x)[0]))

initial_position = jnp.stack(optimized) #(n_chains, n_dim)
# initial_position = kepler_params_ini.T

mean = initial_position.mean(0)
init_centered = (initial_position - mean)
cov = init_centered.T @ init_centered / config['n_chains']

model = RealNVP(10, config['n_dim'], 64, 1)

run_mcmc = jax.vmap(mala_sampler, in_axes=(0, None, None, None, 0, None),
                    out_axes=0)

print("Initializing sampler class")


nf_sampler = Sampler(rng_key_set, config, model, run_mcmc, log_posterior, d_log_posterior)


print("Sampling")

start = time.time()
chains, nf_samples = nf_sampler.sample(initial_position)
print('Elapsed: ', time.time()-start, 's')

print("Make plots")

value1 = true_params
n_dim = config['n_dim']
# Make the base corner plot
figure = corner.corner(chains.reshape(-1,n_dim),
                labels=['v0', 'log_s2', 'log_period', 'log_k', 'sin_phi_', 'cos_phi_', 'ecc_', 'sin_w_', 'cos_w_'])
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
for i in range(np.minimum(config['n_chains'],10)):
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
n_dim = config['n_dim']
# Make the base corner plot
figure = corner.corner(chains.reshape(-1,n_dim),
                labels=['v0', 'log_s2', 'log_period', 'log_k', 'sin_phi_', 'cos_phi_', 'ecc_', 'sin_w_', 'cos_w_'])
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


