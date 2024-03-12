"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, 
                                                     device=device, dtype=dtype))
        self._bias = bias
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, 
                                                       dtype=dtype).transpose())
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        output = X @ self.weight
        if self._bias:
            output = output + self.bias.broadcast_to(output.shape)
        return output
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor):
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        size = np.prod(X.shape).item()
        return X.reshape((batch_size, int(size / batch_size)))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION
    
class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return 1 / (1 + ops.exp(-x))
        ### END YOUR SOLUTION

class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x.tanh()
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        m = logits.shape[0]
        k = logits.shape[1]
        label_onehot = init.one_hot(k, y, device=logits.device, dtype=logits.dtype)
        loss = ops.logsumexp(logits, axes=(1, )) - (logits * label_onehot).sum(axes=(1, ))
        return loss.sum() / m
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        m, n = x.shape
        if self.training:
            mean_x = x.sum(axes=(0, )) / m
            mean_x_broadcast = mean_x.reshape((1, n)).broadcast_to(x.shape)
            var_x = ((x - mean_x_broadcast)**2).sum(axes=(0, )) / m
            var_x_broadcast = var_x.reshape((1, n)).broadcast_to(x.shape)
            try:
                self.running_mean = (1 - self.momentum)*self.running_mean + self.momentum*mean_x.detach()
                self.running_var = (1 - self.momentum)*self.running_var + self.momentum*var_x.detach()
            except AssertionError as e:
                print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
                print(x.shape)
                print(self.running_mean.shape)
                print(mean_x.shape)
                raise e
        else:
            mean_x_broadcast = self.running_mean.reshape((1, n)).broadcast_to(x.shape)
            var_x_broadcast = self.running_var.reshape((1, n)).broadcast_to(x.shape)

        weights_broadcast = self.weight.reshape((1, n)).broadcast_to(x.shape)
        bias_broadcast = self.bias.reshape((1, n)).broadcast_to(x.shape)
        return (x - mean_x_broadcast)/((var_x_broadcast + self.eps) ** 0.5)*weights_broadcast + bias_broadcast
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        _, n = x.shape
        mean_x = x.sum(axes=(1, )) / n
        size = np.prod(mean_x.shape).item()
        mean_x = ops.broadcast_to(mean_x.reshape((size, 1)), x.shape)
        var_x = ((x - mean_x)**2).sum(axes=(1, )) / n
        var_x = ops.broadcast_to(var_x.reshape((size, 1)), x.shape)

        weights_broadcast = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        bias_broadcast = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        return (x - mean_x)/((var_x + self.eps) ** 0.5)*weights_broadcast + bias_broadcast
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*(x.shape), p=1-self.p, 
                              device=x.device, dtype="float32") / (1 - self.p)
            return mask * x
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION

