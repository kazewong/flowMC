from evosax import CMA_ES
import jax
import jax.numpy as jnp
import tqdm

class EvolutionaryOptimizer:

    """
    A wrapper class for the evosax package.
    Note that we do not aim to solve any generic optimization problem,
    especially in a high dimension space.
    """

    def __init__(self, ndims, popsize=100, verbose=False):
        self.strategy = CMA_ES(num_dims=ndims, popsize=popsize, elite_ratio=0.5)
        self.es_params = self.strategy.default_params.replace(clip_min=0, clip_max=1)
        self.verbose = verbose

    def optimize(self, objective, bound, n_loops = 100, seed = 9527):
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
            if self.verbose: progress_bar.set_description(f"Generation: {i}, Fitness: {fitness.mean():.4f}")

    def get_result(self):
        best_member = self.state.best_member* (self.bound[:, 1] - self.bound[:, 0]) + self.bound[:, 0]
        best_fitness = self.state.best_fitness
        return best_member, best_fitness