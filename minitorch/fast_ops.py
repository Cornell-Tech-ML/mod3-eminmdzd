from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT-compiles a function with Numba, applying inline optimization by default.

    Args:
    ----
        fn (Callable): The function to be JIT-compiled.
        **kwargs (Any): Additional keyword arguments to customize the compilation.

    Returns:
    -------
        Callable: The JIT-compiled function with inline optimization.

    """
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


# seems like zip and map need more work
class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        stride_aligned = np.array_equal(out_shape, in_shape) and np.array_equal(
            out_strides, in_strides
        )

        if stride_aligned:
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
        else:
            for ordinal in prange(len(out)):
                # Move the declarations inside the loop
                out_index = np.zeros(len(out_shape), dtype=np.int32)
                in_index = np.zeros(len(in_shape), dtype=np.int32)

                to_index(ordinal, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                out_pos = index_to_position(out_index, out_strides)
                in_pos = index_to_position(in_index, in_strides)
                out[out_pos] = fn(in_storage[in_pos])

    return njit(_map, parallel=True)


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        stride_aligned = (
            np.array_equal(out_shape, a_shape)
            and np.array_equal(out_shape, b_shape)
            and np.array_equal(out_strides, a_strides)
            and np.array_equal(out_strides, b_strides)
        )

        if stride_aligned:
            for i in prange(len(out)):
                out[i] = fn(a_storage[i], b_storage[i])
        else:
            for ordinal in prange(len(out)):
                # Move the declarations inside the loop
                out_index = np.zeros(len(out_shape), dtype=np.int32)
                a_index = np.zeros(len(a_shape), dtype=np.int32)
                b_index = np.zeros(len(b_shape), dtype=np.int32)

                to_index(ordinal, out_shape, out_index)
                broadcast_index(out_index, out_shape, a_shape, a_index)
                broadcast_index(out_index, out_shape, b_shape, b_index)
                out_pos = index_to_position(out_index, out_strides)
                a_pos = index_to_position(a_index, a_strides)
                b_pos = index_to_position(b_index, b_strides)
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return njit(_zip, parallel=True)


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        reduce_size = a_shape[reduce_dim]

        for ordinal in prange(len(out)):
            # Move the declaration inside the loop
            out_index = np.zeros(MAX_DIMS, dtype=np.int32)
            to_index(ordinal, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            acc = out[o]
            for s in range(reduce_size):
                out_index[reduce_dim] = s
                j = index_to_position(out_index, a_strides)
                acc = fn(acc, a_storage[j])
            out[o] = acc

    return njit(_reduce, parallel=True)


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    assert (
        a_shape[-1] == b_shape[-2]
    ), "Incompatible dimensions for matrix multiplication."

    # Determine batch dimensions
    batch_dim = max(len(out_shape) - 2, 0)
    batch_shape = out_shape[:batch_dim]
    batch_size = 1
    for size in batch_shape:
        batch_size *= size
    m, n, k = out_shape[-2], out_shape[-1], a_shape[-1]

    def get_batch_offset(batch_index: int, shape: Shape, strides: Strides) -> int:
        """Calculate the batch offset for a specific batch index in a tensor,
        based on the shape and strides of the tensor.

        Args:
        ----
            batch_index (int): The index of the batch.
            shape (Shape): The shape of the tensor up to the batch dimensions.
            strides (Strides): The strides for the tensor.

        Returns:
        -------
            int: The linear offset in the storage corresponding to the batch index.

        """
        index = np.zeros(len(shape), dtype=np.int32)
        to_index(batch_index, shape, index)
        offset = index_to_position(index, strides)
        return offset

    # Parallelize over batch indices
    for batch_index in prange(batch_size):
        # Compute batch offsets
        a_batch_offset = (
            get_batch_offset(batch_index, a_shape[:batch_dim], a_strides[:batch_dim])
            if len(a_shape) > 2
            else 0
        )
        b_batch_offset = (
            get_batch_offset(batch_index, b_shape[:batch_dim], b_strides[:batch_dim])
            if len(b_shape) > 2
            else 0
        )
        out_batch_offset = (
            get_batch_offset(
                batch_index, out_shape[:batch_dim], out_strides[:batch_dim]
            )
            if len(out_shape) > 2
            else 0
        )

        for i in range(m):
            for j in range(n):
                acc = 0.0
                for kk in range(k):
                    a_pos = a_batch_offset + i * a_strides[-2] + kk * a_strides[-1]
                    b_pos = b_batch_offset + kk * b_strides[-2] + j * b_strides[-1]
                    acc += a_storage[a_pos] * b_storage[b_pos]

                out_pos = out_batch_offset + i * out_strides[-2] + j * out_strides[-1]
                out[out_pos] = acc


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
