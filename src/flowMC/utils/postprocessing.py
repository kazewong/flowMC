import matplotlib.pyplot as plt
import jax.numpy as jnp
# from flowMC.sampler.Sampler import Sampler
from jaxtyping import Float, Array

def plot_summary(sampler: object, training: bool = False, **plotkwargs) -> None:
    """
    Create plots of the most important quantities in the summary.

    Args:
        training (bool, optional): If True, plot training quantities. If False, plot production quantities. Defaults to False.
    """
    
    # Choose the dataset
    data = sampler.get_sampler_state(training=training)
    # TODO add loss values in plotting
    keys = ["local_accs", "global_accs", "log_prob"]
    if sampler.track_gelman_rubin:
        keys.append("gelman_rubin")
    
    # Check if outdir is property of sampler
    if hasattr(sampler, "outdir"):
        outdir = sampler.outdir
    else:
        outdir = "./outdir/"
    
    for key in keys:
        if training:
            which = "training"
        else:
            which = "production"
        _single_plot(data, key, which, outdir=outdir, **plotkwargs)
            
def _single_plot(data: dict, name: str, which: str = "training", outdir: str = "./outdir/", **plotkwargs):
    """
    Create a single plot of a quantity in the summary.

    Args:
        data (dict): Dictionary with the summary data.
        name (str): Name of the quantity to plot.
        which (str, optional): Name of this summary dict. Defaults to "training".
    """
    # Get plot kwargs
    figsize = plotkwargs["figsize"] if "figsize" in plotkwargs else (12, 8)
    alpha = plotkwargs["alpha"] if "alpha" in plotkwargs else 1
    eps = 1e-3
    
    # Prepare plot data
    plotdata = data[name]        
    mean = jnp.mean(plotdata, axis=0)
    x = [i+1 for i in range(len(mean))]
    
    # Plot
    plt.figure(figsize=figsize)
    plt.plot(x, mean, linestyle="-", color="blue", alpha=alpha)
    plt.xlabel("Iteration")
    plt.ylabel(f"{name} ({which})")
    # Extras for some variables:
    if "acc" in name:
        plt.ylim(0-eps, 1+eps)
    plt.savefig(f"{outdir}{name}_{which}.png", bbox_inches='tight')
    
def gelman_rubin(chains: Float[Array, "n_chains n_steps n_dim"], discard_fraction: float = 0.1) -> Array:
    """
    Compute the Gelman-Rubin R statistic for each parameter in the chains.
    """
    _, _, n_dim = jnp.shape(chains)
    
    R_list = []
    
    for i in range(n_dim):
        # Get shape of chains for this parameter
        samples = chains[:, :, i]
        n_chains, length_chain = jnp.shape(samples)
        # Discard burn-in
        start_index = int(jnp.round(discard_fraction * length_chain))
        cut_samples = samples[:, start_index:]
        # Do Gelman-Rubin statistic computation
        chain_means = jnp.mean(cut_samples, axis=1)
        chain_vars = jnp.var(cut_samples, axis=1)
        BoverN = jnp.var(chain_means)
        W = jnp.mean(chain_vars)
        sigmaHat2 = W + BoverN
        m = n_chains
        VHat = sigmaHat2 + BoverN/m
        try:
            R = VHat/W
        except:
            print(f"Error when computer Gelman-Rubin R statistic.")
            R = jnp.nan
        
        R = float(R)
        R_list.append(R)
    
    R_list = jnp.array(R_list)
    return R_list