from flowMC.resource.base import Resource
from flowMC.resource.local_kernel.base import ProposalBase
from flowMC.resource.buffers import Buffer
from flowMC.strategy.base import Strategy
from jaxtyping import Array, Float, PRNGKeyArray
import jax
import equinox as eqx
from typing import Callable
from abc import abstractmethod


class TakeSteps(Strategy):
    logpdf: Callable[[Float[Array, " n_dim"], dict], Float]
    kernel: ProposalBase
    buffer_names: list[str]
    n_steps: int
    current_position: int
    thinning: int
    verbose: bool

    def __init__(
        self,
        logpdf: Callable[[Float[Array, " n_dim"], dict], Float],
        kernel: ProposalBase,
        buffer_names: list[str],
        n_steps: int,
        thinning: int = 1,
        verbose: bool = False,
    ):
        self.logpdf = logpdf
        self.kernel = kernel
        self.buffer_names = buffer_names
        self.n_steps = n_steps
        self.current_position = 0
        self.thinning = thinning
        self.verbose = verbose

    @abstractmethod
    def sample(
        self,
        rng_key: PRNGKeyArray,
        initial_position: Float[Array, " n_dim"],
        data: dict,
    ):
        raise NotImplementedError
    
    def set_current_position(self, current_position: int):
        self.current_position = current_position

    def __call__(
        self,
        rng_key: PRNGKeyArray,
        resources: dict[str, Resource],
        initial_position: Float[Array, "n_chains n_dim"],
        data: dict,
    ) -> tuple[
        PRNGKeyArray,
        dict[str, Resource],
        Float[Array, "n_chains n_dim"],
    ]:
        rng_key, subkey = jax.random.split(rng_key)
        subkey = jax.random.split(subkey, initial_position.shape[0])
        position_buffer = resources[self.buffer_names[0]]
        assert isinstance(
            position_buffer, Buffer
        ), "Position buffer resource must be a Buffer"
        log_prob_buffer = resources[self.buffer_names[1]]
        assert isinstance(
            log_prob_buffer, Buffer
        ), "Log probability buffer resource must be a Buffer"
        acceptance_buffer = resources[self.buffer_names[2]]
        assert isinstance(
            acceptance_buffer, Buffer
        ), "Acceptance buffer resource must be a Buffer"

        # Filter jit will bypass the compilation of the function if not clearing the cache
        positions, log_probs, do_accepts = eqx.filter_jit( 
            eqx.filter_vmap(self.sample, in_axes=(0, 0, None))
        )(subkey, initial_position, data)

        positions = positions[:, :: self.thinning]
        log_probs = log_probs[:, :: self.thinning][..., None]
        do_accepts = do_accepts[:, :: self.thinning][..., None]

        position_buffer.update_buffer(positions, self.n_steps // self.thinning, self.current_position)
        log_prob_buffer.update_buffer(log_probs, self.n_steps // self.thinning, self.current_position)
        acceptance_buffer.update_buffer(do_accepts, self.n_steps // self.thinning, self.current_position)
        self.current_position += self.n_steps // self.thinning
        return rng_key, resources, positions[:,-1]

    def update_kernel(self, kernel: ProposalBase):
        self.kernel = kernel
        eqx.clear_caches() # It would be good to see whether we can get rid of this while recompile correctly


class TakeSerialSteps(TakeSteps):
    
    """
    TakeSerialSteps is a strategy that takes a number of steps in a serial manner, i.e. one after the other.
    This uses jax.lax.scan to iterate over the steps and apply the kernel to the current position.
    This is intended to be used for most local kernels that are dependent on the previous step.
    """

    def body(self, carry, aux):
        key, position, log_prob, data = carry
        position, log_prob, do_accept = self.kernel.kernel(
            key, self.logpdf, position, log_prob, data
        )
        return (key, position, log_prob, data), (position, log_prob, do_accept)

    def sample(
        self,
        rng_key: PRNGKeyArray,
        initial_position: Float[Array, " n_dim"],
        data: dict,
    ):
        (last_key, last_position, last_log_prob, data), (positions, log_probs, do_accepts) = (
            jax.lax.scan(
                self.body,
                (rng_key, initial_position, self.logpdf(initial_position, data), data),
                length=self.n_steps,
            )
        )
        return positions, log_probs, do_accepts

class TakeGroupSteps(TakeSteps):

    """
    TakeGroupSteps is a strategy that takes a number of steps in a group manner, i.e. all steps are taken at once.
    This is intended to be used for kernels such as normalizing flow, which proposal steps are independent of each other,
    and benefit from being computed in parallel.
    """

    def sample(
        self,
        rng_key: PRNGKeyArray,
        initial_position: Float[Array, " n_dim"],
        data: dict,
    ):
        (positions, log_probs, do_accepts) = self.kernel.kernel(
            rng_key, self.logpdf, initial_position, self.logpdf(initial_position, data), {**data, 'n_steps': self.n_steps}
        )
        return positions, log_probs, do_accepts