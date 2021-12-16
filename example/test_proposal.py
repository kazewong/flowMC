
from nfsampler.nfmodel.maf import MaskedAutoregressiveFlow
from nfsampler.nfmodel.realNVP import RealNVP
from nfsampler.sampler.Gaussian_random_walk import rw_metropolis_sampler
from nfsampler.sampler.NF_proposal import nf_metropolis_sampler, nf_metropolis_kernel
import jax
import jax.numpy as jnp                # JAX NumPy
import jax.random as random            # JAX random
from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal
import numpy as np  

from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state
import optax                           # Optimizers
from functools import partial
from jax.scipy.stats import norm


"""
Training a masked autoregressive flow to fit the dual moons dataset.
"""

"""
Define hyper-parameters here.
"""



def train_step(model, state, batch):
	def loss(params):
		y, log_det = model.apply({'params': params},batch)
		mean = jnp.zeros((batch.shape[0],model.n_features))
		cov = jnp.repeat(jnp.eye(model.n_features)[None,:],batch.shape[0],axis=0)
		log_det = log_det + multivariate_normal.logpdf(y,mean,cov)
		return -jnp.mean(log_det)
	grad_fn = jax.value_and_grad(loss)
	value, grad = grad_fn(state.params)
	state = state.apply_gradients(grads=grad)
	return value,state

train_step = jax.jit(train_step,static_argnums=(0,))

def train_flow(rng, model, state, data):

    def train_epoch(state, train_ds, batch_size, epoch, rng):
        """Train for a single epoch."""
        train_ds_size = len(train_ds)
        steps_per_epoch = train_ds_size // batch_size

        perms = jax.random.permutation(rng, train_ds_size)
        perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, batch_size))
        for perm in perms:
            batch = train_ds[perm, ...]
            value, state = train_step(model, state, batch)

        return value, state

    for epoch in range(1, num_epochs + 1):
        print('Epoch %d' % epoch)
        # Use a separate PRNG key to permute image data during shuffling
        rng, input_rng = jax.random.split(rng)
        # Run an optimization step over a training batch
        value, state = train_epoch(state, data, batch_size, epoch, input_rng)
        print('Train loss: %.3f' % value)

    return rng, state

def dual_moon_pe(x):
    """
    Term 2 and 3 separate the distriubiotn and smear it along the first and second dimension
    """
    term1 = 0.5 * ((jnp.linalg.norm(x, axis=-1) - 2) / 0.1) ** 2
    term2 = -0.5 * ((x[..., :1] + jnp.array([-3., 3.])) / 0.8) ** 2
    term3 = -0.5 * ((x[..., 1:2] + jnp.array([-3., 3.])) / 0.6) ** 2
    return -(term1 - logsumexp(term2, axis=-1) - logsumexp(term3, axis=-1))

def sample_nf(model, param, rng_key,n_sample):
    rng_key, subkey = random.split(rng_key)
    samples = model.apply({'params': param}, subkey, n_sample,param, method=model.sample)
    samples = jnp.flip(samples[0],axis=1)
    return rng_key,samples

n_dim = 5
n_samples = 200
nf_samples = 100
n_chains = 100
learning_rate = 0.01
momentum = 0.9
num_epochs = 100
batch_size = 10000
precompiled = False

print("Preparing RNG keys")
rng_key = jax.random.PRNGKey(42)
rng_key_mcmc, rng_key_nf = jax.random.split(rng_key,2)

rng_keys_mcmc = jax.random.split(rng_key_mcmc, n_chains)  # (nchains,)
rng_keys_nf, init_rng_keys_nf = jax.random.split(rng_key_nf,2)

print("Initializing MCMC model and normalizing flow model.")

initial_position = jnp.zeros((n_dim, n_chains)) #(n_dim, n_chains)

#model = MaskedAutoregressiveFlow(n_dim,64,4)
model = RealNVP(10,n_dim,64, 1)
params = model.init(init_rng_keys_nf, jnp.ones((batch_size,n_dim)))['params']

run_mcmc = jax.vmap(rw_metropolis_sampler, in_axes=(0, None, None, 1),
                    out_axes=0)

tx = optax.adam(learning_rate, momentum)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

print("Sampling")


def sampling_loop(rng_keys_nf, rng_keys_mcmc, model, state, initial_position):
	rng_keys_mcmc, positions, log_prob = run_mcmc(rng_keys_mcmc, n_samples, dual_moon_pe, initial_position)
	flat_chain = positions.reshape(-1,n_dim)	
	rng_keys_nf, state = train_flow(rng_key_nf, model, state, flat_chain)	
	rng_keys_nf, nf_chain, log_prob, log_prob_nf = nf_metropolis_sampler(rng_keys_nf, nf_samples, model, state.params , dual_moon_pe, positions[:,-1])

	positions = jnp.concatenate((positions,nf_chain),axis=1)
	return rng_keys_nf, rng_keys_mcmc, state, positions

last_step = initial_position
chains = []
for i in range(3):
	rng_keys_nf, rng_keys_mcmc, state, positions = sampling_loop(rng_keys_nf, rng_keys_mcmc, model, state, last_step)
	last_step = positions[:,-1].T
	# rng_keys_mcmc, positions, log_prob = run_mcmc(rng_keys_mcmc, n_samples, likelihood, initial_position)
	# last_step = last_step.T
	chains.append(positions)
chains = np.concatenate(chains,axis=1)
#nf_samples = sample_nf(model, state.params, rng_keys_nf, 10000)

rng_key, subkey = random.split(rng_keys_nf,2)
proposal_position = model.apply({'params': params}, subkey, initial_position.shape[1]*nf_samples, params, method=model.sample)[0]

log_pdf_nf_proposal = model.apply({'params': params}, proposal_position, method=model.log_prob)
log_pdf_nf_initial = model.apply({'params': params}, initial_position.T, method=model.log_prob)
log_pdf_proposal = dual_moon_pe(proposal_position)
log_pdf_initial = dual_moon_pe(initial_position.T)

proposal_position = proposal_position.reshape(nf_samples, initial_position.shape[1], initial_position.shape[0])
log_pdf_nf_proposal = log_pdf_nf_proposal.reshape(nf_samples, initial_position.shape[1])
log_pdf_proposal = log_pdf_proposal.reshape(nf_samples, initial_position.shape[1])

rng_key, *subkeys = random.split(rng_key,initial_position.shape[1]+1)
subkeys = jnp.array(subkeys)

nf_metropolis_kernel(subkeys, proposal_position[0], initial_position.T, log_pdf_proposal[0], log_pdf_nf_proposal[0], log_pdf_initial, log_pdf_nf_initial)