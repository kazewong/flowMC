from evosax import CMA_ES
import jax
import jax.numpy as jnp
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

    def optimize(self, objective, bound, n_loops = 100, seed = 9527):
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
        progress_bar = tqdm.tqdm(range(n_loops), "Generation: ") if self.verbose else range(n_loops)
        self.bound = bound
        self.state = self.strategy.initialize(rng, self.es_params)
        for i in progress_bar:
            rng, rng_gen, rng_eval = jax.random.split(rng, 3)
            x, state = self.strategy.ask(rng_gen, self.state, self.es_params)
            theta = x * (bound[:, 1] - bound[:, 0]) + bound[:, 0]
            fitness = objective(theta)
            self.state = self.strategy.tell(x, fitness.astype(jnp.float32), state, self.es_params)
            if self.verbose: progress_bar.set_description(f"Generation: {i}, Fitness: {self.state.best_fitness:.4f}")

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

        best_member = self.state.best_member* (self.bound[:, 1] - self.bound[:, 0]) + self.bound[:, 0]
        best_fitness = self.state.best_fitness
        return best_member, best_fitness