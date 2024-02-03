import jax.numpy as jnp

# TODO - add loss values?
def initialize_summary_dict(sampler):
        
    my_dict = dict()

    my_dict["chains"] = jnp.empty((sampler.n_chains, 0, sampler.n_dim))
    my_dict["log_prob"] = jnp.empty((sampler.n_chains, 0))
    my_dict["local_accs"] = jnp.empty((sampler.n_chains, 0))
    my_dict["global_accs"] = jnp.empty((sampler.n_chains, 0))
        
    return my_dict