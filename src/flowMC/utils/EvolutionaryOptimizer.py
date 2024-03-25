from evosax import CMA_ES
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import tqdm


class EvolutionaryOptimizer:
    """
    A wrapper class for the evosax package.
    Note that we do not aim to solve any generic optimization problem,
    especially in a high dimension space.


    Parameters
    ----------
    ndims : int
        The dimension of the parameter space.
    popsize : int
        The population size of the evolutionary algorithm.
    verbose : bool
        Whether to print the progress bar.

    Attributes
    ----------
    strategy : evosax.CMA_ES
        The evolutionary strategy.
    es_params : evosax.CMA_ESParams
        The parameters of the evolutionary strategy.
    verbose : bool
        Whether to print the progress bar.

    Methods
    -------
    optimize(objective, bound, n_loops = 100, seed = 9527)
        Optimize the objective function.
    get_result()
        Get the best member and the best fitness.
    """

    def __init__(self, ndims, popsize=100, verbose=False):
        self.strategy = CMA_ES(num_dims=ndims, popsize=popsize, elite_ratio=0.5)
        self.es_params = self.strategy.default_params.replace(clip_min=0, clip_max=1)
        self.verbose = verbose
        self.history = []
        self.state = None

    def optimize(self, objective, bound, n_loops=100, seed=9527, keep_history_step=0):
        """
        Optimize the objective function.

        Parameters
        ----------
        objective : Callable
            The objective function, which should be implemented in JAX.
        bound : (2, ndims) ndarray
            The bound of the parameter space.
        n_loops : int
            The number of iterations.
        seed : int
            The random seed.

        Returns
        -------
        None
        """
        rng = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(rng)
        progress_bar = (
            tqdm.tqdm(range(n_loops), "Generation: ")
            if self.verbose
            else range(n_loops)
        )
        self.bound = bound
        self.state = self.strategy.initialize(key, self.es_params)
        if keep_history_step > 0:
            self.history = []
            for i in progress_bar:
                subkey, self.state, theta = self.optimize_step(
                    subkey, self.state, objective, bound
                )
                if i % keep_history_step == 0:
                    self.history.append(theta)
                if self.verbose:
                    progress_bar.set_description(
                        f"Generation: {i}, Fitness: {self.state.best_fitness:.4f}"
                    )
            self.history = jnp.array(self.history)
        else:
            for i in progress_bar:
                subkey, self.state, _ = self.optimize_step(
                    subkey, self.state, objective, bound
                )
                if self.verbose:
                    progress_bar.set_description(
                        f"Generation: {i}, Fitness: {self.state.best_fitness:.4f}"
                    )

    def optimize_step(self, key: PRNGKeyArray, state, objective: callable, bound):
        key, subkey = jax.random.split(key)
        x, state = self.strategy.ask(subkey, state, self.es_params)
        theta = x * (bound[:, 1] - bound[:, 0]) + bound[:, 0]
        fitness = objective(theta)
        state = self.strategy.tell(
            x, fitness.astype(jnp.float32), state, self.es_params
        )
        return key, state, theta

    def get_result(self):
        """
        Get the best member and the best fitness.

        Returns
        -------
        best_member : (ndims,) ndarray
            The best member.
        best_fitness : float
            The best fitness.
        """

        best_member = (
            self.state.best_member * (self.bound[:, 1] - self.bound[:, 0])
            + self.bound[:, 0]
        )
        best_fitness = self.state.best_fitness
        return best_member, best_fitness
