# Utilities for using and be used by autoinf.
from typing import Type

from nnsmith.abstract.op import *


def make_reduce_type(axis) -> Type[ReduceBase]:
    class Reduce(ReduceBase):
        def _init_reduce_dim(self, input_shape):
            self.extra_attrs["reduce_dim"] = axis
            return self.extra_attrs["reduce_dim"]

    return Reduce


ATTR_FREE_RULES = [
    ElementWiseUnaryOp,
    BcastBinaryOp,
    Where,
    *[make_reduce_type(i) for i in range(__MAX_RANK__)],
    Tril,
    Triu,
    Concat1,
    Concat2,
    Concat3,
    Concat4,
    Concat5,
    MatMul,
]

"""
~ Number of input operands:
```
OP_TYPE.n_input()
```
~ Number of output operands:
```
OP_TYPE.n_output()
```
~ Shape transfer rule:
```
op = OP_TYPE()
List of AbsTensor = op.checked_type_transfer(List of AbsTensor(shape=..., dtype=...))
# CHECK: Output shape same.
```
~ Input constraint rule:
```
op = OP_TYPE()
List of predicates = op.checked_requires(List of AbsTensor(shape=..., dtype=...))
# CHECK: All true.
```
"""
