from flowMC.resource.base import Resource
from flowMC.resource.local_kernel.base import ProposalBase
from flowMC.resource.buffers import Buffer
from flowMC.resource.logPDF import TemperedPDF
from flowMC.strategy.base import Strategy
from jaxtyping import Array, Float, PRNGKeyArray, Int, Bool
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
    verbose: bool = False

    def __init__(
        self,
        n_steps: int,
        tempered_logpdf_name: str,
        kernel_name: str,
        tempered_buffer_names: list[str],
        verbose: bool = False,
    ):
        self.n_steps = n_steps
        self.tempered_logpdf_name = tempered_logpdf_name
        self.kernel_name = kernel_name
        self.tempered_buffer_names = tempered_buffer_names
        self.verbose = verbose

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
        """
        Resources must contain:
            - TemperedPDF
            - Local kernel
            - A buffer holding the tempered positions
            - A buffer holding the temperatures

        This strategy has 3 main steps:
        1. Sample from the tempered PDF using the local kernel for n_steps
        2. Exchange the samples between the temperatures
        3. Adapt the temperatures based on the acceptance rate

        TODO: Add way to turn of temperature adaptation to maintain detail balance.
        """

        rng_key, subkey = jax.random.split(rng_key)
        kernel = resources[self.kernel_name]
        assert isinstance(kernel, ProposalBase)
        tempered_logpdf = resources[self.tempered_logpdf_name]
        assert isinstance(tempered_logpdf, TemperedPDF)
        tempered_positions = resources[
            self.tempered_buffer_names[0]
        ]  # Shape (n_chains, n_temps, n_dims)
        assert isinstance(tempered_positions, Buffer)
        temperatures = resources[self.tempered_buffer_names[1]]
        assert isinstance(temperatures, Buffer)

        initial_position = jnp.concatenate(
            [initial_position[:, None, :], tempered_positions.data],
            axis=1,
        )

        # Take individual steps

        rng_key, subkey = jax.random.split(rng_key)
        subkey = jax.random.split(subkey, initial_position.shape[0])
        positions, log_probs, do_accepts = eqx.filter_jit(
            eqx.filter_vmap(
                jax.tree_util.Partial(self._ensemble_step, kernel),
                in_axes=(0, 0, None, None, None),
            )
        )(
            subkey, initial_position, tempered_logpdf, temperatures.data, data
        )  # vmapping over chains

        # Exchange between temperatures

        rng_key, subkey = jax.random.split(rng_key)
        subkey = jax.random.split(subkey, initial_position.shape[0])
        positions, log_probs, do_accepts = eqx.filter_jit(
            eqx.filter_vmap(self._exchange, in_axes=(0, 0, 0, None, None))
        )(subkey, positions, tempered_logpdf, temperatures.data, data)

        # Update the buffers

        tempered_positions.update_buffer(positions[:, 1:], 0)

        # Adapt the temperatures
        temperatures.update_buffer(
            eqx.filter_jit(self._adapt_temperature)(temperatures.data, do_accepts), 0
        )

        return rng_key, resources, positions[:, 0]

    def _individual_step_body(
        self,
        kernel: ProposalBase,
        carry: tuple[
            PRNGKeyArray,
            Float[Array, " n_dims"],
            Float[Array, "1"],
            TemperedPDF,
            Float[Array, " n_temps"],
            dict,
        ],
        aux,
    ) -> tuple[
        tuple[
            PRNGKeyArray,
            Float[Array, " n_dims"],
            Float[Array, "1"],
            TemperedPDF,
            Float[Array, " n_temps"],
            dict,
        ],
        tuple[
            Float[Array, " n_dims"],
            Float[Array, "1"],
            Int[Array, "1"],
        ],
    ]:
        """Take a step using the kernel and the tempered logpdf.
        This should not be called directly but instead used in a
        jax.lax.scan to take multiple steps.

        Args:
            kernel (ProposalBase): The kernel to use.
            carry (tuple): The current state of the chain.
                - key (PRNGKeyArray): jax random key.
                - position (Float[Array, " n_dims"]): Current position of the chain.
                - log_prob (Float[Array, "1"]): Current log probability of the chain.
                - logpdf (TemperedPDF): The tempered LogPDF class.
                - temperatures (Float[Array, " n_temps"]): Array of temperatures.
                - data (dict): Additional data to pass to the logpdf.
            aux (None): Not used.
        Returns:
            tuple: Updated carry and the result of the kernel step.
                - carry (tuple): Updated state of the chain.
                    - key (PRNGKeyArray): jax random key.
                    - position (Float[Array, " n_dims"]): New position of the chain.
                    - log_prob (Float[Array, "1"]): New log probability of the chain.
                    - logpdf (TemperedPDF): The tempered LogPDF class.
                    - temperatures (Float[Array, " n_temps"]): Array of temperatures.
                    - data (dict): Additional data to pass to the logpdf.
                - result (tuple): Result of the kernel step.
                    - position (Float[Array, " n_dims"]): New position of the chain.
                    - log_prob (Float[Array, "1"]): New log probability of the chain.
                    - do_accept (Int[Array, "1"]): Whether the new position is accepted.
        """
        key, position, log_prob, logpdf, temperatures, data = carry
        key, subkey = jax.random.split(key)
        position, log_prob, do_accept = kernel.kernel(
            subkey,
            position,
            log_prob,
            jax.tree_util.Partial(logpdf.tempered_log_pdf, temperatures),
            data,
        )
        return (key, position, log_prob, logpdf, temperatures, data), (
            position,
            log_prob,
            do_accept,
        )

    def _individal_step(
        self,
        kernel: ProposalBase,
        rng_key: PRNGKeyArray,
        positions: Float[Array, " n_dims"],
        logpdf: TemperedPDF,
        temperatures: Float[Array, " n_temps"],
        data: dict,
    ) -> tuple[
        Float[Array, " n_dims"],
        Float[Array, "1"],
        Int[Array, "1"],
    ]:
        """
        Perform a series of individual steps for a single chain using the kernel.

        Args:
            kernel (ProposalBase): The kernel to use for proposing new positions.
            rng_key (PRNGKeyArray): jax random key for reproducibility.
            positions (Float[Array, " n_dims"]): Current positions of the chain.
            logpdf (TemperedPDF): The tempered log probability density function.
            temperatures (Float[Array, " n_temps"]): Array of temperatures.
            data (dict): Additional data to pass to the logpdf.

        Returns:
            tuple:
                - positions (Float[Array, " n_dims"]): Updated positions of the chain.
                - log_probs (Float[Array, "1"]): Log probabilities of the chain.
                - do_accept (Int[Array, "1"]): Acceptance flag for the new position.
        """
        log_probs = logpdf(positions, data)

        (key, position, log_prob, logpdf, temperatures, data), (
            positions,
            log_probs,
            do_accept,
        ) = jax.lax.scan(
            jax.tree_util.Partial(self._individual_step_body, kernel),
            ((rng_key, positions, log_probs, logpdf, temperatures, data)),
            length=self.n_steps,
        )
        return position, log_prob, do_accept

    def _ensemble_step(
        self,
        kernel: ProposalBase,
        rng_key: PRNGKeyArray,
        positions: Float[Array, "n_temps n_dims"],
        logpdf: TemperedPDF,
        temperatures: Float[Array, " n_temps"],
        data: dict,
    ) -> tuple[
        Float[Array, "n_temps n_dims"],
        Float[Array, " n_temps"],
        Int[Array, " n_temps"],
    ]:
        """
        Perform ensemble steps for all chains and temperatures.

        Args:
            kernel (ProposalBase): The kernel to use for proposing new positions.
            rng_key (PRNGKeyArray): Random key for reproducibility.
            positions (Float[Array, "n_temps n_dims"]): Current positions for all temperatures.
            logpdf (TemperedPDF): The tempered log probability density function.
            temperatures (Float[Array, " n_temps"]): Array of temperatures.
            data (dict): Additional data to pass to the logpdf.

        Returns:
            tuple:
                - positions (Float[Array, "n_temps n_dims"]): Updated positions for all temperatures.
                - log_probs (Float[Array, " n_temps"]): Log probabilities for all temperatures.
                - do_accept (Int[Array, " n_temps"]): Acceptance flags for each temperature.
        """

        if self.verbose:
            print("Taking individual steps")
        rng_key = jax.random.split(rng_key, positions.shape[0])

        positions, log_probs, do_accept = jax.vmap(
            self._individal_step, in_axes=(None, 0, 0, None, 0, None)
        )(kernel, rng_key, positions, logpdf, temperatures, data)

        return positions, log_probs, do_accept

    def _exchange_step_body(
        self,
        carry: tuple[
            PRNGKeyArray,
            Float[Array, "n_temps n_dims"],
            Float[Array, " n_temps"],
            int,
            TemperedPDF,
            Float[Array, " n_temps"],
            dict,
        ],
        aux: None,
    ) -> tuple[
        tuple[
            PRNGKeyArray,
            Float[Array, "n_temps n_dims"],
            Float[Array, " n_temps"],
            int,
            TemperedPDF,
            Float[Array, " n_temps"],
            dict,
        ],
        Int[Array, "1"],
    ]:

        key, positions, log_probs, idx, logpdf, temperatures, data = carry

        key, subkey = jax.random.split(key)
        ratio = (1.0 / temperatures[idx + 1] - 1.0 / temperatures[idx]) * (
            log_probs[idx] - log_probs[idx + 1]
        )
        log_uniform = jnp.log(jax.random.uniform(subkey))
        do_accept: Bool[Array, " 1"] = log_uniform < ratio
        swapped = jnp.flip(jax.lax.dynamic_slice_in_dim(positions, idx, 2, axis=0))
        positions = jax.lax.cond(
            do_accept,
            true_fun=lambda: jax.lax.dynamic_update_slice_in_dim(
                positions, swapped, idx, axis=0
            ),
            false_fun=lambda: positions,
        )
        swapped_log_probs = jax.vmap(logpdf, in_axes=(0, None))(swapped, data)
        log_probs = jax.lax.cond(
            do_accept,
            true_fun=lambda: jax.lax.dynamic_update_slice_in_dim(
                log_probs, swapped_log_probs, idx, axis=0
            ),
            false_fun=lambda: log_probs,
        )
        return (key, positions, log_probs, idx, logpdf, temperatures, data), do_accept

    def _exchange(
        self,
        key: PRNGKeyArray,
        positions: Float[Array, "n_temps n_dims"],
        logpdf: TemperedPDF,
        temperatures: Float[Array, " n_temps"],
        data: dict,
    ) -> tuple[
        Float[Array, "n_temps n_dims"],
        Float[Array, " n_temps"],
        Int[Array, " n_temps - 1"],
    ]:
        """
        Perform exchange steps between adjacent temperatures.

        Args:
            key (PRNGKeyArray): jax random key for reproducibility.
            positions (Float[Array, "n_temps n_dims"]): Current positions for all temperatures.
            logpdf (TemperedPDF): The tempered log probability density function.
            temperatures (Float[Array, " n_temps"]): Array of temperatures.
            data (dict): Additional data to pass to the logpdf.

        Returns:
            tuple:
                - positions (Float[Array, "n_temps n_dims"]): Updated positions for all temperatures.
                - log_probs (Float[Array, " n_temps"]): Log probabilities for all temperatures.
                - do_accept (Int[Array, " n_temps - 1"]): Acceptance flags for each temperature.
        """

        if self.verbose:
            print("Exchanging walkers")

        log_probs = jax.vmap(logpdf, in_axes=(0, None))(positions, data)
        (key, positions, log_probs, idx, logpdf, temperatures, data), do_accept = (
            jax.lax.scan(
                self._exchange_step_body,
                (key, positions, log_probs, 0, logpdf, temperatures, data),
                length=positions.shape[0] - 1,
            )
        )
        return positions, log_probs, do_accept

    def _adapt_temperature(
        self,
        temperatures: Float[Array, " n_temps"],
        do_accept: Int[Array, " n_chains n_temps 1"],
    ) -> Float[Array, " n_temps"]:
        """
        Adapt the temperatures based on the acceptance rates.

        Args:
            temperatures (Float[Array, " n_temps"]): Current temperatures.
            do_accept (Int[Array, " n_chains n_temps 1"]): Acceptance flags for each chain and temperature.

        Returns:
            Float[Array, " n_temps"]: Updated temperatures.

        TODO: The adaptation now let's the temperature to go above the maximum temperature.
        Need to add a check to prevent this.
        """
        # Adapt the temperature based on the acceptance rate

        if self.verbose:
            print("Adapting temperatures")

        acceptance_rate = jnp.mean(do_accept, axis=0)
        damping_factor = acceptance_rate[:-1] - acceptance_rate[1:]
        new_temperatures = temperatures
        for i in range(1, temperatures.shape[0] - 1):
            new_temperatures = new_temperatures.at[i].set(
                new_temperatures[i - 1]
                + (temperatures[i] - temperatures[i - 1]) * jnp.exp(damping_factor[i])
            )

        return new_temperatures
