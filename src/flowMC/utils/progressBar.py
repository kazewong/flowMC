from tqdm import tqdm
from jax import jit, vmap, grad, random, lax
import jax.numpy as jnp
from jax.experimental import host_callback


def progress_bar_scan(num_samples, message=None):
    "Progress bar for a JAX scan"
    if message is None:
        message = f"Running for {num_samples:,} iterations"
    tqdm_bars = tqdm(range(num_samples))
    tqdm_bars.set_description(message)

    if num_samples > 20:
        print_rate = int(num_samples / 20)
    else:
        print_rate = 1  # if you run the sampler for less than 20 iterations

    print_rate = 1
    remainder = num_samples % print_rate

    def _define_tqdm(arg, transform):
        tqdm_bars = tqdm(range(num_samples))
        tqdm_bars.set_description(message)

    def _update_tqdm(arg, transform):
        tqdm_bars.update(arg)

    def _update_progress_bar(iter_num):
        "Updates tqdm progress bar of a JAX scan or loop"
        _ = lax.cond(
            iter_num == 0,
            lambda _: host_callback.id_tap(_define_tqdm, None, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = lax.cond(
            # update tqdm every multiple of `print_rate` except at the end
            (iter_num % print_rate == 0) & (iter_num != num_samples - remainder),
            lambda _: host_callback.id_tap(_update_tqdm, print_rate, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = lax.cond(
            # update tqdm by `remainder`
            iter_num == num_samples - remainder,
            lambda _: host_callback.id_tap(_update_tqdm, remainder, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

    def _close_tqdm(arg, transform):
        tqdm_bars.close()

    def close_tqdm(result, iter_num):
        return lax.cond(
            iter_num == num_samples - 1,
            lambda _: host_callback.id_tap(_close_tqdm, None, result=result),
            lambda _: result,
            operand=None,
        )

    def _progress_bar_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `lax.scan`.
        Note that `body_fun` must either be looping over `np.arange(num_samples)`,
        or be looping over a tuple who's first element is `np.arange(num_samples)`
        This means that `iter_num` is the current iteration number
        """

        def wrapper_progress_bar(iter_num, state):
            _update_progress_bar(iter_num)
            result = func(iter_num, state)
            return close_tqdm(result, iter_num)

        return wrapper_progress_bar

    return _progress_bar_scan


# @progress_bar_scan(10000)
# def loop_body(i, s):
#   s += 1
#   return s
