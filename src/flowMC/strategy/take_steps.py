from flowMC.resource.base import Resource
from flowMC.resource.log_pdf import LogPDF
from flowMC.resource.local_kernel.base import ProposalBase
from flowMC.resource.buffers import Buffer
from flowMC.strategy.base import Strategy
from jaxtyping import Array, Float, PRNGKeyArray
import jax
import equinox as eqx

class TakeLocalSteps(Strategy):

    kernel_resource: str
    buffer_resource: str
    n_steps: int
    thinning: int
    verbose: bool

    def __str__(self):
        return "Take " + str(self.n_steps) + " local steps with " + self.kernel_resource

    def __init__(
        self,
        kernel_resource: str,
        buffer_resource: str,
        n_steps: int,
        thinning: int = 1,
        verbose: bool = False,
    ):
        self.kernel_resource = kernel_resource
        self.buffer_resource = buffer_resource
        self.n_steps = n_steps
        self.thinning = thinning
        self.verbose = verbose

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
        logpdf = resources['LogPDF']
        assert isinstance(logpdf, LogPDF), "LogPDF resource must be a LogPDF"
        logpdf = logpdf.logpdf
        kernel = resources[self.kernel_resource]
        assert isinstance(kernel, ProposalBase), "Kernel resource must be a ProposalBase"
        buffer = resources[self.buffer_resource]
        assert isinstance(buffer, Buffer), "Buffer resource must be a Buffer"

        def body(carry, data):
            key, position, log_prob = carry
            position, log_prob, do_accept = kernel.kernel(key, logpdf, position, log_prob, data)
            return (key, position, log_prob), (position, log_prob, do_accept)

        def sample(rng_key, initial_position, data):
            (last_key, last_position, last_log_prob), (positions, log_probs, do_accepts) = jax.lax.scan(body, (rng_key, initial_position, logpdf(initial_position, data)), length=self.n_steps)
            return positions, log_probs, do_accepts

        positions, log_probs, do_accepts = eqx.filter_jit(sample)(rng_key, initial_position, data)

        buffer.update_buffer
  
        return rng_key, resources, position
