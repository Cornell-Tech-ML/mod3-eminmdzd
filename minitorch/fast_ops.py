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
    """Function njit"""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


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
        stride_aligned = (
            len(out_shape) == len(in_shape)
            and np.array_equal(out_shape, in_shape)
            and len(out_strides) == len(in_strides)
            and np.array_equal(out_strides, in_strides)
        )
        total_size = len(out)

        # If stride-aligned, avoid indexing calculations
        if stride_aligned:
            for i in prange(total_size):
                out[i] = fn(in_storage[i])
            return

        all_indices = np.zeros((total_size, 2, MAX_DIMS), np.int32)
        for i in prange(total_size):
            to_index(i, out_shape, all_indices[i, 0])
            broadcast_index(all_indices[i, 0], out_shape, in_shape, all_indices[i, 1])
            in_ordinal = index_to_position(all_indices[i, 1], in_strides)
            out[i] = fn(in_storage[in_ordinal])

    return njit(_map, parallel=True)  # type: ignore


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
            (len(out_shape) == len(a_shape) == len(b_shape))
            and np.array_equal(out_shape, a_shape)
            and np.array_equal(out_shape, b_shape)
            and (len(out_strides) == len(a_strides) == len(b_strides))
            and np.array_equal(out_strides, a_strides)
            and np.array_equal(out_strides, b_strides)
        )
        total_size = len(out)

        # If stride-aligned, avoid indexing calculations
        if stride_aligned:
            for i in prange(total_size):
                out[i] = fn(a_storage[i], b_storage[i])
            return

        all_indices = np.zeros((total_size, 3, MAX_DIMS), np.int32)

        for i in prange(total_size):
            to_index(i, out_shape, all_indices[i, 0])

            broadcast_index(all_indices[i, 0], out_shape, a_shape, all_indices[i, 1])
            a_ordinal = index_to_position(all_indices[i, 1], a_strides)

            broadcast_index(all_indices[i, 0], out_shape, b_shape, all_indices[i, 2])
            b_ordinal = index_to_position(all_indices[i, 2], b_strides)

            out[i] = fn(a_storage[a_ordinal], b_storage[b_ordinal])

    return njit(_zip, parallel=True)  # type: ignore


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
        reduce_step = a_strides[reduce_dim]
        total_size = len(out)
        out_index_array = np.zeros((total_size, MAX_DIMS), np.int32)

        for i in prange(total_size):
            to_index(i, out_shape, out_index_array[i])
            a_ordinal_start = index_to_position(out_index_array[i], a_strides)

            current_ordinal = a_ordinal_start + 0
            acc = out[i]
            for _ in range(reduce_size):
                acc = fn(acc, a_storage[current_ordinal])
                current_ordinal += reduce_step
            out[i] = acc

    return njit(_reduce, parallel=True)  # type: ignore


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
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    o_batch_stride = out_strides[0]
    batch_count = out_shape[0]

    for batch_i in prange(batch_count):
        o_batch_offset = batch_i * o_batch_stride
        a_batch_offset = batch_i * a_batch_stride
        b_batch_offset = batch_i * b_batch_stride

        for a_row_i in range(a_shape[-2]):
            a_row_offset = a_batch_offset + a_row_i * a_strides[-2]
            o_row_offset = o_batch_offset + a_row_i * out_strides[-2]
            for b_col_j in range(b_shape[-1]):
                b_col_offset = b_batch_offset + b_col_j * b_strides[-1]
                o_col_offset = b_col_j * out_strides[-1]
                dot_sum = 0.0
                for k in range(a_shape[-1]):
                    a_idx = a_row_offset + k * a_strides[-1]
                    b_idx = b_col_offset + k * b_strides[-2]
                    dot_sum += a_storage[a_idx] * b_storage[b_idx]
                out[o_row_offset + o_col_offset] = dot_sum


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
