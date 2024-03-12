"""Operator implementations."""

from numbers import Number
from typing import Optional, Sequence, Tuple, Union

from ..autograd import NDArray
from ..autograd import Tensor, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from ..backend_selection import array_api
from .ops_tuple import *

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray): 
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * self.scalar * (a ** (self.scalar - 1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad: Tensor, node: Tensor):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a: NDArray, b: NDArray):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad / b
        grad_b = -out_grad * a / (b ** 2)
        return grad_a, grad_b
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if axes != None and len(axes) != 2:
            raise ValueError("Axes must be None or tuple of size 2")
        self.axes = axes

    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        ndim = a.ndim
        if ndim < 2:
            return a
        new_axes = list(range(ndim))
        if self.axes is None:
            self.axes = tuple(range(ndim)[-2:])
        (new_axes[self.axes[0]], 
         new_axes[self.axes[1]]) = (new_axes[self.axes[1]], 
                                    new_axes[self.axes[0]])
        return a.permute(new_axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return a.compact().reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        origin_shape = a.shape
        return out_grad.reshape(origin_shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return a.broadcast_to(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        origin_shape = a.shape
        origin_shape_pad = (1, ) * (len(self.shape) - len(origin_shape)) + origin_shape
        axes = tuple(i for i, (size_from, size_to) in enumerate(zip(origin_shape_pad, self.shape)) 
                     if size_from != size_to)
        grad = summation(out_grad, axes=axes)
        if grad.shape == origin_shape:
            return grad
        else:
            return grad.reshape(origin_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[Union[int, Tuple[int, ...]]] = None):
        if isinstance(axes, Sequence):
            self.axes = tuple(axes)
        elif isinstance(axes, int):
            self.axes = (axes, )
        elif axes is None:
            self.axes = (None, )
        else:
            raise ValueError(f"Do not support axes of type `{type(axes)}`")

    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        for axis in self.axes:
            a = a.sum(axis=axis)
        return a.squeeze()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        origin_shape = a.shape
        origin_ndim = len(origin_shape)
        out_shape = list(origin_shape)
        if self.axes[0] is None:
            out_shape_pad = tuple([1]*origin_ndim)
        else:
            for axis in self.axes:
                out_shape[axis] = 1
            out_shape_pad = tuple(out_shape)
        temp = reshape(out_grad, out_shape_pad)
        grad = broadcast_to(temp, origin_shape)
        return grad
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad @ transpose(b)
        grad_b = transpose(a) @ out_grad
        return grad_a, grad_b
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return a.log()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad / a
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return a.exp()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return out_grad * node
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return a.maximum(0.0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        real_data = node.realize_cached_data()
        mask_array = real_data > 0
        mask_tensor = Tensor(mask_array, device=out_grad.device, dtype=out_grad.dtype)
        return out_grad * mask_tensor
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return a.tanh()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return out_grad * (1 - node**2) 
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: Tuple[NDArray, ...]) -> NDArray:
        ### BEGIN YOUR SOLUTION
        data_num = len(args)
        shape = list(args[0].shape)
        shape.insert(self.axis, data_num)
        device = args[0].device
        dtype = args[0].dtype
        out = array_api.empty(shape, dtype=dtype, device=device)

        shape_item = shape.copy()
        shape_item[self.axis] = 1
        idx = [slice(None, None) for _ in range(len(shape))]
        for i, item in enumerate(args):
            idx[self.axis] = i
            out[tuple(idx)] = item.compact().reshape(shape_item)
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A: NDArray) -> Tuple[NDArray, ...]:
        ### BEGIN YOUR SOLUTION
        shape = A.shape
        out = []

        shape_item = list(shape)
        shape_item.pop(self.axis)
        idx = [slice(None, None) for _ in range(len(shape))]
        for i in range(shape[self.axis]):
            idx[self.axis] = i
            data = A[tuple(idx)].compact().reshape(shape_item)
            out.append(data)
        return tuple(out)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: TensorTuple, node: TensorTuple):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, axis=self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        shape = list(a.shape)
        idx = [slice(None, None) for _ in range(len(shape))]
        for axis in self.axes:
            shape[axis] *= 1 + self.dilation 
            idx[axis] = slice(None, None, 1+self.dilation)
        out = a.device.full(shape, 0, dtype=a.dtype)
        out[tuple(idx)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        idx = [slice(None, None) for _ in range(len(a.shape))]
        for axis in self.axes:
            idx[axis] = slice(None, None, 1+self.dilation)
        return a[tuple(idx)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A: NDArray, B:NDArray):
        ### BEGIN YOUR SOLUTION
        assert A.ndim == 4 and B.ndim == 4
        if self.padding > 0:
            pad_axes = ((0, 0), *((self.padding,)*2, )*2, (0, 0))
            A = A.pad(pad_axes)

        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        assert H >= K and W >= K, "kernel size must le than H and W"
        H_out, W_out = int((H - K)/self.stride) + 1, int((W - K)/self.stride) + 1 

        A_ = A.compact().as_strided((N, H_out, W_out, K, K, C_in), 
                                    (H * W * C_in, self.stride * W * C_in, 
                                     self.stride * C_in, W * C_in, C_in, 1))
        A_ = A_.compact().reshape((N * H_out * W_out, K * K * C_in))
        B_ = B.compact().reshape((K * K * C_in, C_out))
        out = (A_ @ B_).compact().reshape((N, H_out, W_out, C_out))
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs
        K = B.shape[0]
        
        if self.stride > 1:
            out_grad_dilate = dilate(out_grad, (1, 2), dilation=self.stride-1)
        else:
            out_grad_dilate = out_grad

        B_transpose = transpose(flip(B, (0, 1)), (2, 3))
        assert K - 1- self.padding >= 0, "do not support neg padding now"
        grad_A = conv(out_grad_dilate, B_transpose, stride=1, padding=K-1-self.padding)

        A_transpose = transpose(A, (0, 3)) # C_in, H, W, N
        kernel = transpose(transpose(out_grad_dilate, (0, 1)), (1, 2)) # H_out, W_out, N, C_out
        grad_B = conv(A_transpose, kernel, stride=1, padding=self.padding) # C_in, K, K, C_out
        grad_B = transpose(transpose(grad_B, (0, 1)), (1, 2))
        return grad_A, grad_B
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=0):
    return Conv(stride, padding)(a, b)
