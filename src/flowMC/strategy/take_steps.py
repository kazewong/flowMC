from flowMC.resource.base import Resource
from flowMC.resource.kernel.base import ProposalBase
from flowMC.resource.buffers import Buffer
from flowMC.resource.states import State
from flowMC.resource.logPDF import LogPDF
from flowMC.strategy.base import Strategy
from jaxtyping import Array, Float, PRNGKeyArray
import jax
import jax.numpy as jnp
import equinox as eqx
from abc import abstractmethod


class TakeSteps(Strategy):
    logpdf_name: str
    kernel_name: str
    state_name: str
    buffer_names: list[str]
    n_steps: int
    current_position: int
    thinning: int
    chain_batch_size: int  # If vmap over a large number of chains is memory bounded, this splits the computation
    verbose: bool

    def __init__(
        self,
        logpdf_name: str,
        kernel_name: str,
        state_name: str,
        buffer_names: list[str],
        n_steps: int,
        thinning: int = 1,
        chain_batch_size: int = 0,
        verbose: bool = False,
    ):
        self.logpdf_name = logpdf_name
        self.kernel_name = kernel_name
        self.state_name = state_name
        self.buffer_names = buffer_names
        self.n_steps = n_steps
        self.current_position = 0
        self.thinning = thinning
        self.chain_batch_size = chain_batch_size
        self.verbose = verbose

    @abstractmethod
    def sample(
        self,
        kernel: ProposalBase,
        rng_key: PRNGKeyArray,
        initial_position: Float[Array, " n_dim"],
        logpdf: LogPDF,
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

        assert isinstance(
            state_resource := resources[self.state_name], State
        ), "State resource must be a State"

        assert isinstance(
            position_buffer_name := state_resource.data[self.buffer_names[0]], str
        ), "Position buffer resource name must be a string"

        assert isinstance(
            log_prob_buffer_name := state_resource.data[self.buffer_names[1]], str
        ), "Log probability buffer resource name must be a string"

        assert isinstance(
            acceptance_buffer_name := state_resource.data[self.buffer_names[2]], str
        ), "Acceptance buffer resource name must be a string"

        assert isinstance(
            position_buffer := resources[position_buffer_name], Buffer
        ), "Position buffer resource must be a Buffer"
        assert isinstance(
            log_prob_buffer := resources[log_prob_buffer_name], Buffer
        ), "Log probability buffer resource must be a Buffer"
        assert isinstance(
            acceptance_buffer := resources[acceptance_buffer_name], Buffer
        ), "Acceptance buffer resource must be a Buffer"

        kernel = resources[self.kernel_name]
        logpdf = resources[self.logpdf_name]

        # Filter jit will bypass the compilation of
        # the function if not clearing the cache
        n_chains = initial_position.shape[0]
        if self.chain_batch_size > 1 and n_chains > self.chain_batch_size:
            positions_list = []
            log_probs_list = []
            do_accepts_list = []
            for i in range(0, n_chains, self.chain_batch_size):
                batch_slice = slice(i, min(i + self.chain_batch_size, n_chains))
                subkey_batch = subkey[batch_slice]
                initial_position_batch = initial_position[batch_slice]
                positions_batch, log_probs_batch, do_accepts_batch = eqx.filter_jit(
                    eqx.filter_vmap(
                        jax.tree_util.Partial(self.sample, kernel),
                        in_axes=(0, 0, None, None),
                    )
                )(subkey_batch, initial_position_batch, logpdf, data)
                positions_list.append(positions_batch)
                log_probs_list.append(log_probs_batch)
                do_accepts_list.append(do_accepts_batch)
            positions = jnp.concatenate(positions_list, axis=0)
            log_probs = jnp.concatenate(log_probs_list, axis=0)
            do_accepts = jnp.concatenate(do_accepts_list, axis=0)
        else:
            positions, log_probs, do_accepts = eqx.filter_jit(
                eqx.filter_vmap(
                    jax.tree_util.Partial(self.sample, kernel),
                    in_axes=(0, 0, None, None),
                )
            )(subkey, initial_position, logpdf, data)

        positions = positions[:, :: self.thinning]
        log_probs = log_probs[:, :: self.thinning]
        do_accepts = do_accepts[:, :: self.thinning].astype(
            acceptance_buffer.data.dtype
        )

        position_buffer.update_buffer(positions, self.current_position)
        log_prob_buffer.update_buffer(log_probs, self.current_position)
        acceptance_buffer.update_buffer(do_accepts, self.current_position)
        self.current_position += self.n_steps // self.thinning
        return rng_key, resources, positions[:, -1]


class TakeSerialSteps(TakeSteps):
    """TakeSerialSteps is a strategy that takes a number of steps in a serial manner,
    i.e. one after the other.

    This uses jax.lax.scan to iterate over the steps and apply the kernel to the current
    position. This is intended to be used for most local kernels that are dependent on
    the previous step.
    """

    def body(self, kernel: ProposalBase, carry, aux):
        key, position, log_prob, logpdf, data = carry
        key, subkey = jax.random.split(key)
        position, log_prob, do_accept = kernel.kernel(
            subkey, position, log_prob, logpdf, data
        )
        return (key, position, log_prob, logpdf, data), (position, log_prob, do_accept)

    def sample(
        self,
        kernel: ProposalBase,
        rng_key: PRNGKeyArray,
        initial_position: Float[Array, " n_dim"],
        logpdf: LogPDF,
        data: dict,
    ):
        (
            (last_key, last_position, last_log_prob, logpdf, data),
            (positions, log_probs, do_accepts),
        ) = jax.lax.scan(
            jax.tree_util.Partial(self.body, kernel),
            (rng_key, initial_position, logpdf(initial_position, data), logpdf, data),
            length=self.n_steps,
        )
        return positions, log_probs, do_accepts


class TakeGroupSteps(TakeSteps):
    """TakeGroupSteps is a strategy that takes a number of steps in a group manner, i.e.
    all steps are taken at once.

    This is intended to be used for kernels such as normalizing flow, which proposal
    steps are independent of each other, and benefit from being computed in parallel.
    """

    def sample(
        self,
        kernel: ProposalBase,
        rng_key: PRNGKeyArray,
        initial_position: Float[Array, " n_dim"],
        logpdf: LogPDF,
        data: dict,
    ):
        (positions, log_probs, do_accepts) = kernel.kernel(
            rng_key,
            initial_position,
            logpdf(initial_position, data),
            logpdf,
            {**data, "n_steps": self.n_steps},
        )
        return positions, log_probs, do_accepts
