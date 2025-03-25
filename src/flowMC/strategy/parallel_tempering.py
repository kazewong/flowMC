from flowMC.resource.base import Resource
from flowMC.resource.local_kernel.base import ProposalBase
from flowMC.resource.buffers import Buffer
from flowMC.resource.logPDF import TemperedPDF
from flowMC.strategy.base import Strategy
from jaxtyping import Array, Float, PRNGKeyArray, Int
import jax
import jax.numpy as jnp
import equinox as eqx


class ParallelTempering(Strategy):
    """Sample a tempered PDF with one exchange step.
    This is in essence closer to TakeSteps than global tuning.
    Considering the tempered version of the PDF is only there to
    help with convergence, by default the extra information in
    temperature not equal to 1 is not saved.

    There should be a version of this class that saves the extra
    information in the temperature not equal to 1, which could be
    used for other purposes such as diagnostics or training.


    """

    n_steps: int
    tempered_logpdf_name: str
    kernel_name: str
    tempered_buffer_names: list[str]

    def __init__(
        self,
        n_steps: int,
        tempered_logpdf_name: str,
        kernel_name: str,
        tempered_buffer_names: list[str],
    ):
        self.n_steps = n_steps
        self.tempered_logpdf_name = tempered_logpdf_name
        self.kernel_name = kernel_name
        self.tempered_buffer_names = tempered_buffer_names

    def __call__(
        self,
        rng_key: PRNGKeyArray,
        resources: dict[str, Resource],
        initial_position: Float[Array, "n_chains n_dims"],
        data: dict,
    ) -> tuple[
        PRNGKeyArray,
        dict[str, Resource],
        Float[Array, "n_chains n_dim"],
    ]:

        rng_key, subkey = jax.random.split(rng_key)
        kernel = resources[self.kernel_name]
        assert isinstance(kernel, ProposalBase)
        tempered_logpdf = resources[self.tempered_logpdf_name]
        assert isinstance(tempered_logpdf, TemperedPDF)
        tempered_positions = resources[
            self.tempered_buffer_names[0]
        ]  # Shape (n_chains, n_temps, n_dims)
        assert isinstance(tempered_positions, Buffer)

        # Take individual steps

        rng_key, subkey = jax.random.split(rng_key)
        positions, log_probs, do_accepts = eqx.filter_jit(
            eqx.filter_vmap(
                jax.tree_util.Partial(self._individal_step, kernel),
                in_axes=(0, 0, None),
            )
        )(
            subkey, tempered_positions.data, tempered_logpdf, data
        )  # vmapping over chains

        # Exchange between temperatures

        final_position = self._exchange(positions, kernel, tempered_logpdf)

        return rng_key, resources, final_position[0]

    def _step_body(
        self,
        kernel: ProposalBase,
        carry: tuple[
            PRNGKeyArray,
            Float[Array, "n_temps n_dims"],
            Float[Array, "n_temps 1"],
            TemperedPDF,
            dict,
        ],
    ) -> tuple[
        tuple[
            PRNGKeyArray,
            Float[Array, "n_temps n_dims"],
            Float[Array, "n_temps 1"],
            TemperedPDF,
            dict,
        ],
        tuple[
            Float[Array, "n_temps n_dims"],
            Float[Array, "n_temps 1"],
            Int[Array, "n_temps 1"],
        ],
    ]:
        key, position, log_prob, logpdf, data = carry
        key, subkey = jax.random.split(key)
        position, log_prob, do_accept = jax.vmap(
            kernel.kernel, in_axes=(0, 0, 0, None, None)
        )(subkey, position, log_prob, logpdf, data)
        return (key, position, log_prob, logpdf, data), (position, log_prob, do_accept)

    def _individal_step(
        self,
        kernel: ProposalBase,
        rng_key: PRNGKeyArray,
        positions: Float[Array, "n_temps n_dims"],
        logpdf: TemperedPDF,
        data: dict,
    ):
        (
            rng_key,
            positions,
            log_prob,
            logpdf,
            data,
        ), (positions, log_prob, do_accept) = jax.lax.scan(
            jax.tree_util.Partial(self._step_body, kernel),
            (rng_key, positions, logpdf(positions, data), logpdf, data),
        )
        return positions, log_prob, do_accept

    def _exchange(
        self,
        positions: Float[Array, "n_temps n_dims"],
        kernel: ProposalBase,
        logpdf: TemperedPDF,
    ):
        raise NotImplementedError
        # log_prob_temps = jax.vmap(logpdf, in_axes=(0, None))(
        #     positions, None
        # ) * evaluate_temps(temps_pdf, positions, temperatures)
