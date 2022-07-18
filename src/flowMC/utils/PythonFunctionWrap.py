from functools import wraps
from typing import Any, Callable, List, Tuple, Dict, Iterable, NamedTuple, Union

Array = Any
PyTree = Union[Array, Iterable[Array], Dict[Any, Array], NamedTuple]
SampleStats = Dict[str, Array]
Extras = PyTree


import jax
import jax.numpy as jnp
import numpy as np
from jax._src import dtypes
from jax.custom_batching import custom_vmap
from jax.experimental import host_callback
from jax.tree_util import tree_flatten



def wrap_python_log_prob_fn(
    python_log_prob_fn: Callable[..., Array]
) -> LogProbFn:
    @custom_vmap
    @wraps(python_log_prob_fn)
    def log_prob_fn(params: Array) -> Array:
        dtype = _tree_dtype(params)
        return host_callback.call(
            python_log_prob_fn,
            params,
            result_shape=jax.ShapeDtypeStruct((), dtype),
        )

    @log_prob_fn.def_vmap
    def _(
        axis_size: int, in_batched: List[bool], params: Array
    ) -> Tuple[Array, bool]:
        del axis_size, in_batched

        if _arraylike(params):
            flat_params = params
            eval_one = python_log_prob_fn
        else:
            flat_params, unravel = ravel_ensemble(params)
            eval_one = lambda x: python_log_prob_fn(unravel(x))

        result_shape = jax.ShapeDtypeStruct(
            (flat_params.shape[0],), flat_params.dtype
        )
        return (
            host_callback.call(
                lambda y: np.stack([eval_one(x) for x in y]),
                flat_params,
                result_shape=result_shape,
            ),
            True,
        )

    return log_prob_fn


def _tree_dtype(tree: PyTree) -> Any:
    leaves, _ = tree_flatten(tree)
    from_dtypes = [dtypes.dtype(l) for l in leaves]
    return dtypes.result_type(*from_dtypes)


def _arraylike(x: Array) -> bool:
    return (
        isinstance(x, np.ndarray)
        or isinstance(x, jnp.ndarray)
        or hasattr(x, "__jax_array__")
        or np.isscalar(x)
    )