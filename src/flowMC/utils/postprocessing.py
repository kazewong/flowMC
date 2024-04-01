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
