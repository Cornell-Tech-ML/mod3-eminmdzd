# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
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

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:  # noqa: ANN003
    """JIT-compile a function to run on CUDA device."""
    return cuda.jit(device=True, **kwargs)(fn)


def jit(fn: Callable[..., Any], **kwargs: Any) -> FakeCUDAKernel:
    """Compile a function to run as a CUDA kernel."""
    return cuda.jit(**kwargs)(fn)


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
        """See `tensor_ops.py`"""
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
        """See `tensor_ops.py`"""
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
        """See `tensor_ops.py`"""
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
) -> Callable[[Storage, Shape, Strides, int, Storage, Shape, Strides], None]:
    """CUDA tensor map function to apply a single-argument function element-wise on a tensor.

    Args:
    ----
        fn (Callable[[float], float]): Function to apply element-wise.

    Returns:
    -------
        Callable: The CUDA kernel function for element-wise mapping.

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
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i >= out_size:
            return

        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)

        to_index(i, out_shape, out_index)
        broadcast_index(out_index, out_shape, in_shape, in_index)

        out_pos = index_to_position(out_index, out_strides)
        in_pos = index_to_position(in_index, in_strides)
        out[out_pos] = fn(in_storage[in_pos])

    return cuda.jit()(_map)


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, int, Storage, Shape, Strides, Storage, Shape, Strides],
    None,
]:
    """CUDA tensor zip function to apply a two-argument function element-wise on two tensors.

    Args:
    ----
        fn (Callable[[float, float], float]): Function to apply element-wise on pairs.

    Returns:
    -------
        Callable: The CUDA kernel function for element-wise zip operation.

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
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i >= out_size:
            return

        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)

        to_index(i, out_shape, out_index)
        broadcast_index(out_index, out_shape, a_shape, a_index)
        broadcast_index(out_index, out_shape, b_shape, b_index)

        out_pos = index_to_position(out_index, out_strides)
        a_pos = index_to_position(a_index, a_strides)
        b_pos = index_to_position(b_index, b_strides)
        out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """CUDA kernel to perform block-wise summation of elements in `a` into `out`, using shared memory.

    Each block sums its portion of `a` and writes the result to the corresponding element in `out`.

    Args:
    ----
        out (Storage): Storage for output tensor.
        a (Storage): Storage for input tensor.
        size (int): Length of `a`.

    """
    BLOCK_DIM = 32
    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # Load into shared memory
    cache[pos] = a[i] if i < size else 0.0
    cuda.syncthreads()

    # Perform reduction within the block
    step = 1
    while step < BLOCK_DIM:
        if pos % (2 * step) == 0 and pos + step < BLOCK_DIM:
            cache[pos] += cache[pos + step]
        step *= 2
        cuda.syncthreads()

    # Write block result to global memory
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Perform block-wise summation of elements in the input tensor `a` using the CUDA kernel `_sum_practice`.

    This function sets up the necessary grid and block configuration for the CUDA kernel to
    sum the elements of `a` in chunks of `THREADS_PER_BLOCK` and stores the partial sums in
    the output tensor `out`. Each block in the grid computes the sum of a segment of `a`,
    writing one partial result per block.

    Args:
    ----
        a (Tensor): Input tensor to be summed. Assumed to be a 1D tensor.

    Returns:
    -------
        TensorData: A tensor containing the partial sums from each block.

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
) -> Callable[
    [Storage, Shape, Strides, int, Storage, Shape, Strides, int, float], None
]:
    """CUDA tensor reduce function that reduces elements along a specified dimension.

    Args:
    ----
        fn (Callable[[float, float], float]): Reduction function.

    Returns:
    -------
        Callable: The CUDA kernel function for reduction along a dimension.

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

        # Calculate position and load into shared memory
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        if i < out_size:
            to_index(i, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)
            cache[cuda.threadIdx.x] = reduce_value
            for j in range(a_shape[reduce_dim]):
                out_index[reduce_dim] = j
                a_pos = index_to_position(out_index, a_strides)
                cache[cuda.threadIdx.x] = fn(cache[cuda.threadIdx.x], a_storage[a_pos])

        cuda.syncthreads()

        # Perform reduction in shared memory
        step = 1
        while step < BLOCK_DIM:
            if cuda.threadIdx.x % (2 * step) == 0:
                cache[cuda.threadIdx.x] = fn(
                    cache[cuda.threadIdx.x], cache[cuda.threadIdx.x + step]
                )
            step *= 2
            cuda.syncthreads()

        # Write result to global memory
        if cuda.threadIdx.x == 0:
            out[out_pos] = cache[0]

    return jit(_reduce)


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Practice CUDA kernel for matrix multiplication of two small matrices (size < 32).

    This function computes the product of matrices `a` and `b`, both of which are of shape `[size, size]`,
    and stores the result in `out`. The matrices are loaded into shared memory for optimized access,
    ensuring each element is read only once and written to global memory once.

    Args:
    ----
        out (Storage): Storage for the output matrix.
        a (Storage): Storage for the input matrix `a`.
        b (Storage): Storage for the input matrix `b`.
        size (int): The size of the matrices (assumed to be a square matrix).

    """
    BLOCK_DIM = 32  # Assume size < 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Compute global indices
    row = cuda.threadIdx.x
    col = cuda.threadIdx.y

    # Load elements of `a` and `b` into shared memory
    if row < size and col < size:
        a_shared[row, col] = a[row * size + col]
        b_shared[row, col] = b[row * size + col]
    else:
        # Pad out-of-bounds indices with zero
        a_shared[row, col] = 0.0
        b_shared[row, col] = 0.0

    cuda.syncthreads()

    # Initialize accumulator for output element
    acc = 0.0
    if row < size and col < size:
        # Compute the dot product for this element
        for k in range(size):
            acc += a_shared[row, k] * b_shared[k, col]

    # Write the result to the output matrix
    if row < size and col < size:
        out[row * size + col] = acc


jit_mm_practice = cuda.jit()(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Perform a practice matrix multiplication of two small square matrices `a` and `b` using the CUDA kernel `_mm_practice`.

    This function configures the grid and block dimensions to run `_mm_practice`, a CUDA kernel that multiplies
    two matrices assumed to be of shape `[size, size]`, where `size` is less than or equal to `THREADS_PER_BLOCK`.
    The kernel performs matrix multiplication using shared memory to minimize global memory accesses.

    Args:
    ----
        a (Tensor): The first input matrix, assumed to be square and of shape `[size, size]`.
        b (Tensor): The second input matrix, also assumed to be square and of shape `[size, size]`.

    Returns:
    -------
        TensorData: The result of the matrix multiplication, a tensor of shape `[size, size]`.

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


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
    """CUDA tensor matrix multiply function.

    Uses shared memory to compute matrix multiplication with batching support.
    Each thread computes one element of the resulting matrix.

    Args:
    ----
        out (Storage): Output storage.
        out_shape (Shape): Shape of the output matrix.
        out_strides (Strides): Strides of the output matrix.
        out_size (int): Size of the output matrix.
        a_storage (Storage): Storage for matrix `a`.
        a_shape (Shape): Shape of matrix `a`.
        a_strides (Strides): Strides of matrix `a`.
        b_storage (Storage): Storage for matrix `b`.
        b_shape (Shape): Shape of matrix `b`.
        b_strides (Strides): Strides of matrix `b`.

    """
    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Batch index for matrices
    batch = cuda.blockIdx.z
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    acc = 0.0
    for kk in range(0, a_shape[-1], BLOCK_DIM):
        if i < a_shape[-2] and kk + pj < a_shape[-1]:
            a_pos = batch * a_strides[0] + i * a_strides[-2] + (kk + pj) * a_strides[-1]
            a_shared[pi, pj] = a_storage[a_pos]
        else:
            a_shared[pi, pj] = 0.0

        if kk + pi < b_shape[-2] and j < b_shape[-1]:
            b_pos = batch * b_strides[0] + (kk + pi) * b_strides[-2] + j * b_strides[-1]
            b_shared[pi, pj] = b_storage[b_pos]
        else:
            b_shared[pi, pj] = 0.0

        cuda.syncthreads()

        # Compute product
        for k in range(BLOCK_DIM):
            acc += a_shared[pi, k] * b_shared[k, pj]
        cuda.syncthreads()

    # Write result
    if i < out_shape[-2] and j < out_shape[-1]:
        out_pos = batch * out_strides[0] + i * out_strides[-2] + j * out_strides[-1]
        out[out_pos] = acc


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
