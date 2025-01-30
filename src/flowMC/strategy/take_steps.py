from flowMC.resource.base import Resource
from flowMC.resource.local_kernel.base import ProposalBase
from flowMC.resource.buffers import Buffer
from flowMC.strategy.base import Strategy
from jaxtyping import Array, Float, PRNGKeyArray
import jax
import equinox as eqx
from typing import Callable

class TakeLocalSteps(Strategy):

    logpdf: Callable[[Float[Array, " n_dim"], dict], Float]
    kernel: ProposalBase
    buffer_name: str
    n_steps: int
    thinning: int
    verbose: bool

    def __repr__(self):
        return "Take " + str(self.n_steps) + " local steps with " + self.kernel.__str__() + " kernel"

    def __init__(
        self,
        logpdf: Callable[[Float[Array, " n_dim"], dict], Float],
        kernel: ProposalBase,
        buffer: str,
        n_steps: int,
        thinning: int = 1,
        verbose: bool = False,
    ):
        self.logpdf = logpdf
        self.kernel = kernel
        self.buffer = buffer
        self.n_steps = n_steps
        self.thinning = thinning
        self.verbose = verbose

    def body(self, carry, data):
        key, position, log_prob = carry
        position, log_prob, do_accept = self.kernel.kernel(key, self.logpdf, position, log_prob, data)
        return (key, position, log_prob), (position, log_prob, do_accept)

    def sample(self, rng_key, initial_position, data):
        (last_key, last_position, last_log_prob), (positions, log_probs, do_accepts) = jax.lax.scan(self.body, (rng_key, initial_position, self.logpdf(initial_position, data)), length=self.n_steps)
        return positions, log_probs, do_accepts


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
        position_buffer = resources[self.buffer + '_position']
        assert isinstance(position_buffer, Buffer), "Position buffer resource must be a Buffer"
        log_prob_buffer = resources[self.buffer + '_log_prob']
        assert isinstance(log_prob_buffer, Buffer), "Log probability buffer resource must be a Buffer"
        acceptance_buffer = resources[self.buffer + '_acceptance']
        assert isinstance(acceptance_buffer, Buffer), "Acceptance buffer resource must be a Buffer"
        
        positions, log_probs, do_accepts = eqx.filter_jit(eqx.filter_vmap(self.sample, in_axes=(0, 0, None)))(rng_key, initial_position, data)

        positions = positions[:, :: self.thinning]
        log_probs = log_probs[:, :: self.thinning][...,None]
        do_accepts = do_accepts[:, :: self.thinning][...,None]

        position_buffer.update_buffer(positions, self.n_steps // self.thinning)  
        log_prob_buffer.update_buffer(log_probs, self.n_steps // self.thinning)
        acceptance_buffer.update_buffer(do_accepts, self.n_steps // self.thinning)
        return rng_key, resources, positions

    def update_kernel(self, kernel: ProposalBase):
        self.kernel = kernel