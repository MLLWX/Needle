from typing import Optional, Sequence
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[Union[int, Sequence[int]]] = None):
        if isinstance(axes, Sequence):
            self.axes = tuple(axes)
        elif isinstance(axes, int):
            self.axes = (axes, )
        elif axes is None:
            self.axes = (None, )
        else:
            raise ValueError(f"Do not support axes of type `{type(axes)}`")

    def compute(self, Z: NDArray):
        ### BEGIN YOUR SOLUTION
        for axis in self.axes:
            max_z = Z.max(axis=axis)
            max_z_ = max_z.broadcast_to(Z.shape)
            Z = (Z - max_z_).exp().sum(axis=axis).log()
            Z += max_z
        return Z.squeeze()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        if self.axes[0] is None:
            shape = [1] * len(Z.shape)
        else:
            shape = list(Z.shape)
            for axis in self.axes:
                shape[axis] = 1
        node_broadcast = broadcast_to(node.reshape(shape), Z.shape)
        softmax_prob = (Z - node_broadcast).exp()
        grad = broadcast_to(out_grad.reshape(shape), Z.shape) * softmax_prob
        return grad
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

