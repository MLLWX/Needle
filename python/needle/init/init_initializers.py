import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    shape = kwargs.pop("shape", None)
    if shape is None:
        shape = (fan_in, fan_out)
    weights = rand(*shape, low=-a, high=a, **kwargs)
    return weights
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    shape = kwargs.pop("shape", None)
    if shape is None:
        shape = (fan_in, fan_out)
    weights = randn(*shape, mean=0, std=std, **kwargs)
    return weights
    ### END YOUR SOLUTION


def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = math.sqrt(2)
    bound = gain * math.sqrt(3 / fan_in)
    shape = kwargs.pop("shape", None)
    if shape is None:
        shape = (fan_in, fan_out)
    return rand(*shape, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = math.sqrt(2)
    std = gain / math.sqrt(fan_in)
    shape = kwargs.pop("shape", None)
    if shape is None:
        shape = (fan_in, fan_out)
    return randn(*shape, mean=0, std=std, **kwargs)
    ### END YOUR SOLUTION