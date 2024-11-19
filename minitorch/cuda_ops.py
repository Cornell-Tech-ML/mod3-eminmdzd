# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions of your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:  # noqa: ANN003
    """Wrapper for JIT compilation to the device for CUDA operations."""
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn, **kwargs) -> FakeCUDAKernel:  # noqa: ANN001, ANN003
    """JIT wrapper for CUDA kernels."""
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Apply a binary function `fn` element-wise to two tensors."""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Apply a reduction function `fn` along specified dimension."""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Perform matrix multiplication on tensors `a` and `b`."""
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

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

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
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Implement for Task 3.3.
        if i >= out_size:
            return
        to_index(i, out_shape, out_index)
        broadcast_index(out_index, out_shape, in_shape, in_index)
        in_pos = index_to_position(in_index, in_strides)
        out_pos = index_to_position(out_index, out_strides)
        out[out_pos] = fn(numba.float64(in_storage[in_pos]))

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::
      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i >= out_size:
            return

        to_index(i, out_shape, out_index)
        broadcast_index(out_index, out_shape, a_shape, a_index)
        broadcast_index(out_index, out_shape, b_shape, b_index)

        a_pos = index_to_position(a_index, a_strides)
        b_pos = index_to_position(b_index, b_strides)
        out_pos = index_to_position(out_index, out_strides)

        out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""A practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # Implement for Task 3.3.
    # Load data into shared memory
    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = numba.float64(0.0)
    cuda.syncthreads()

    # Perform reduction in shared memory
    stride = BLOCK_DIM // 2
    while stride > 0:
        if pos < stride:
            cache[pos] += cache[pos + stride]
        cuda.syncthreads()
        stride //= 2

    # Write the result for this block to out
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Practice kernel to perform summation of tensor elements using shared memory in CUDA.

    Given an array of length `n` and output of size `n // blockDim`,
    it will sum up elements within each block and write to output.

    Args:
    ----
        out (Storage): Storage for output tensor.
        a (Storage): Storage for input tensor.
        size (int): Length of input tensor `a`.

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # Implement for Task 3.3.
        if out_pos >= out_size:
            return

        # Initialize cache with reduce_value
        cache[pos] = reduce_value

        # Compute the index into the output tensor
        to_index(out_pos, out_shape, out_index)

        # Copy to a_index
        for d in range(len(out_shape)):
            a_index[d] = out_index[d]

        reduce_dim_size = a_shape[reduce_dim]

        # Loop over the dimension being reduced
        for s in range(pos, reduce_dim_size, BLOCK_DIM):
            a_index[reduce_dim] = s
            a_pos = index_to_position(a_index, a_strides)
            cache[pos] = fn(numba.float64(cache[pos]), a_storage[a_pos])

        cuda.syncthreads()

        # Perform reduction in shared memory
        stride = BLOCK_DIM // 2
        while stride > 0:
            if pos < stride:
                cache[pos] = fn(cache[pos], cache[pos + stride])
            cuda.syncthreads()
            stride //= 2

        # Write the result to the output tensor
        if pos == 0:
            out_position = index_to_position(out_index, out_strides)
            out[out_position] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32  # Define the block dimension for CUDA kernel
    # Declare shared memory arrays for `a` and `b`
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Thread indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Load 'a' and 'b' into shared memory
    if tx < size and ty < size:
        a_shared[ty, tx] = a[
            ty * size + tx
        ]  # Copy element from global to shared memory for `a`
        b_shared[ty, tx] = b[
            ty * size + tx
        ]  # Copy element from global to shared memory for `b`
    else:
        a_shared[ty, tx] = numba.float64(
            0.0
        )  # Zero padding for elements outside matrix
        b_shared[ty, tx] = numba.float64(
            0.0
        )  # Zero padding for elements outside matrix
    cuda.syncthreads()  # Synchronize all threads in the block

    # Each thread computes one element of the output matrix
    if tx < size and ty < size:
        s = numba.float64(0.0)  # Initialize accumulator
        for k in range(size):  # Perform the dot product for row `i` and column `j`
            s += a_shared[ty, k] * b_shared[k, tx]
        out[ty * size + tx] = s  # Write result to global memory


jit_mm_practice = jit(_mm_practice)  # JIT compile the CUDA kernel


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Practice kernel for matrix multiplication using shared memory in CUDA.

    This kernel performs matrix multiplication for two square matrices `a` and `b` with a size smaller than 32.

    Args:
    ----
        out (Storage): Storage for output matrix.
        a (Storage): Storage for input matrix `a`.
        b (Storage): Storage for input matrix `b`.
        size (int): Size of the square matrices.

    """
    (size, _) = a.shape  # Get matrix dimensions (assumes square matrices)
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)  # Define block size
    blockspergrid = 1  # Define grid size (single grid for small matrices)
    out = TensorData(
        [0.0 for i in range(size * size)], (size, size)
    )  # Initialize output tensor
    out.to_cuda_()  # Move output tensor to CUDA device memory
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )  # Launch the kernel
    return out  # Return the resulting matrix


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function with optimized memory access.
    Handles broadcasting and batch matrix multiplication correctly.
    """
    BLOCK_DIM = 32  # Define block dimension

    # Shared memory arrays - one read per cell
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Thread indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Block indices
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    batch = cuda.blockIdx.z  # Batch index for batched matrix multiplication

    # Global indices
    row = bx * BLOCK_DIM + tx  # Row index in global memory
    col = by * BLOCK_DIM + ty  # Column index in global memory

    # Matrix dimensions
    M = out_shape[-2]  # Rows of A and output
    N = out_shape[-1]  # Cols of B and output
    K = a_shape[-1]  # Cols of A, rows of B

    # Handle broadcasting for batched matrix multiplication
    a_batches = a_shape[0] if len(a_shape) > 2 else 1  # Number of batches in `a`
    b_batches = b_shape[0] if len(b_shape) > 2 else 1  # Number of batches in `b`

    # Get actual batch index with broadcasting
    a_batch_idx = batch % a_batches  # Batch index for `a`
    b_batch_idx = batch % b_batches  # Batch index for `b`

    # Batch strides (0 if not batched)
    a_batch = a_strides[0] if len(a_shape) > 2 else 0
    b_batch = b_strides[0] if len(b_shape) > 2 else 0
    out_batch = out_strides[0] if len(out_shape) > 2 else 0

    # Initialize accumulator in register
    acc = numba.float64(0.0)

    # Process the matrix in BLOCK_DIM x BLOCK_DIM tiles
    for k_tile in range(0, K, BLOCK_DIM):  # Iterate over tiles in K dimension
        # Load tile from A into shared memory
        if row < M and (k_tile + ty) < K:
            # Calculate position in A with broadcasting
            a_idx = (
                a_batch_idx * a_batch
                + row * a_strides[-2]
                + (k_tile + ty) * a_strides[-1]
            )  # Index in global memory for `a`
            a_shared[tx, ty] = a_storage[
                a_idx
            ]  # Copy data from global to shared memory
        else:
            a_shared[tx, ty] = 0.0  # Zero padding for elements outside matrix

        # Load tile from B into shared memory
        if (k_tile + tx) < K and col < N:
            # Calculate position in B with broadcasting
            b_idx = (
                b_batch_idx * b_batch
                + (k_tile + tx) * b_strides[-2]
                + col * b_strides[-1]
            )  # Index in global memory for `b`
            b_shared[tx, ty] = b_storage[
                b_idx
            ]  # Copy data from global to shared memory
        else:
            b_shared[tx, ty] = 0.0  # Zero padding for elements outside matrix

        # Make sure all threads have loaded their data
        cuda.syncthreads()

        # Compute partial dot product for this tile
        if row < M and col < N:
            # Multiply and accumulate using shared memory
            for k in range(min(BLOCK_DIM, K - k_tile)):
                acc += a_shared[tx, k] * b_shared[k, ty]

        # Synchronize before loading next tile
        cuda.syncthreads()

    # Write result to global memory - only once per thread
    if row < M and col < N:
        out_idx = batch * out_batch + row * out_strides[-2] + col * out_strides[-1]
        out[out_idx] = acc  # Store accumulated result in global memory


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
