from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

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
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        maxz = array_api.max(Z, axis=self.axes, keepdims=True)
        out = array_api.log(array_api.sum(array_api.exp(Z - maxz), axis=self.axes)) + array_api.max(Z, axis=self.axes)
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        diff_axes = self.axes if self.axes else tuple(range(len(Z.shape)))
        expansion = [dim if axis not in diff_axes else 1 for axis, dim in enumerate(Z.shape)]
        new_node = node.reshape(shape=expansion).broadcast_to(shape=Z.shape)
        new_grad = out_grad.reshape(shape=expansion).broadcast_to(shape=Z.shape)
        return new_grad * exp(Z - new_node)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

