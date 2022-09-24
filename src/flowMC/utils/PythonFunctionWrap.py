import warnings
from functools import wraps
from typing import Any, List, Dict, Callable, Iterable, Tuple, NamedTuple, Union

Array = Any
PyTree = Union[Array, Iterable[Array], Dict[Any, Array], NamedTuple]
SampleStats = Dict[str, Array]
Extras = PyTree

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src import dtypes
from jax._src.util import safe_zip
from jax.custom_batching import custom_vmap
from jax.experimental import host_callback
from jax.tree_util import tree_flatten, tree_unflatten


def wrap_python_log_prob_fn(python_log_prob_fn: Callable[..., Array]):
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
    def _(axis_size: int, in_batched: List[bool], params: Array) -> Tuple[Array, bool]:
        del axis_size, in_batched

        if _arraylike(params):
            flat_params = params
            eval_one = python_log_prob_fn
        else:
            flat_params, unravel = ravel_ensemble(params)
            eval_one = lambda x: python_log_prob_fn(unravel(x))

        result_shape = jax.ShapeDtypeStruct((flat_params.shape[0],), flat_params.dtype)

        result = host_callback.call(
            lambda y: np.stack([eval_one(x) for x in y]),
            flat_params,
            result_shape=result_shape,
        )
        return (
            result,
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


UnravelFn = Callable[[Array], PyTree]

zip = safe_zip


def ravel_ensemble(coords: PyTree) -> Tuple[Array, UnravelFn]:
    leaves, treedef = tree_flatten(coords)
    flat, unravel_inner = _ravel_inner(leaves)
    unravel_one = lambda flat: tree_unflatten(treedef, unravel_inner(flat))
    return flat, unravel_one


def _ravel_inner(lst: List[Array]) -> Tuple[Array, UnravelFn]:
    if not lst:
        return jnp.array([], jnp.float32), lambda _: []
    from_dtypes = [dtypes.dtype(l) for l in lst]
    to_dtype = dtypes.result_type(*from_dtypes)
    shapes = [jnp.shape(x)[1:] for x in lst]
    indices = np.cumsum([int(np.prod(s)) for s in shapes])

    if all(dt == to_dtype for dt in from_dtypes):
        del from_dtypes, to_dtype

        def unravel(arr: Array) -> PyTree:
            chunks = jnp.split(arr, indices[:-1])
            return [chunk.reshape(shape) for chunk, shape in zip(chunks, shapes)]

        ravel = lambda arg: jnp.concatenate([jnp.ravel(e) for e in arg])
        raveled = jax.vmap(ravel)(lst)
        return raveled, unravel

    else:

        def unravel(arr: Array) -> PyTree:
            arr_dtype = dtypes.dtype(arr)
            if arr_dtype != to_dtype:
                raise TypeError(
                    f"unravel function given array of dtype {arr_dtype}, "
                    f"but expected dtype {to_dtype}"
                )
            chunks = jnp.split(arr, indices[:-1])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return [
                    lax.convert_element_type(chunk.reshape(shape), dtype)
                    for chunk, shape, dtype in zip(chunks, shapes, from_dtypes)
                ]

        ravel = lambda arg: jnp.concatenate(
            [jnp.ravel(lax.convert_element_type(e, to_dtype)) for e in arg]
        )
        raveled = jax.vmap(ravel)(lst)
        return raveled, unravel
