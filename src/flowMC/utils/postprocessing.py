import os

import jax.numpy as jnp
import matplotlib.pyplot as plt

from flowMC import Sampler


def plot_summary(sampler: Sampler, **plotkwargs) -> None:
    """
    Create plots of the most important quantities in the summary.

    Args:
        training (bool, optional): If True, plot training quantities. If False, plot production quantities. Defaults to False.
    """
    keys = ["local_accs", "global_accs", "log_prob"]

    # Check if outdir is property of sampler
    if hasattr(sampler, "outdir"):
        outdir = sampler.outdir
    else:
        outdir = "./outdir/"

    if outdir[-1] != "/":
        outdir += "/"

    os.makedirs(outdir, exist_ok=True)

    training_sampler_state = sampler.get_sampler_state(training=True)

    _loss_val_plot(training_sampler_state["loss_vals"], outdir=outdir, **plotkwargs)

    production_sampler_state = sampler.get_sampler_state(training=False)

    for key in keys:
        training_data = training_sampler_state[key]
        production_data = production_sampler_state[key]
        _stacked_plot(
            training_data,
            production_data,
            key,
            outdir=outdir,
            **plotkwargs,
        )


def _stacked_plot(
    training_data: dict,
    production_data: dict,
    name: str,
    outdir: str = "./outdir/",
    **plotkwargs,
):
    training_data_mean = jnp.mean(training_data, axis=0)
    production_data_mean = jnp.mean(production_data, axis=0)
    x_training = list(range(1, len(training_data_mean) + 1))
    x_production = list(range(1, len(production_data_mean) + 1))

    figsize = plotkwargs.get("figsize", (15, 10))
    alpha = plotkwargs.get("alpha", 1)
    eps = 1e-3

    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True, sharey=True)
    ax[0].plot(
        x_training, training_data_mean, linestyle="-", color="#3498DB", alpha=alpha
    )
    ax[1].plot(
        x_production, production_data_mean, linestyle="-", color="#3498DB", alpha=alpha
    )
    ax[0].set_ylabel(f"{name} (training)")
    ax[1].set_ylabel(f"{name} (production)")
    plt.xlabel("Iteration")
    if "acc" in name:
        plt.ylim(0 - eps, 1 + eps)
    plt.savefig(f"{outdir}{name}.png", bbox_inches="tight")


def _loss_val_plot(
    data,
    outdir: str = "./outdir/",
    **plotkwargs,
):
    # Get plot kwargs
    figsize = plotkwargs["figsize"] if "figsize" in plotkwargs else (12, 8)
    alpha = plotkwargs["alpha"] if "alpha" in plotkwargs else 1

    data_to_plot = data.reshape(-1)
    x = list(range(1, len(data_to_plot) + 1))

    # Plot
    plt.figure(figsize=figsize)
    plt.plot(x, data_to_plot, linestyle="-", color="#3498DB", alpha=alpha)
    plt.xlabel("Iteration")
    plt.ylabel("loss")
    # Extras for some variables:
    plt.savefig(f"{outdir}loss.png", bbox_inches="tight")
