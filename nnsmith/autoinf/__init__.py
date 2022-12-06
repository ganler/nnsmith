# Utilities for using and be used by autoinf.
from typing import Type

from nnsmith.abstract.op import *


def make_reduce_type(axis) -> Type[ReduceBase]:
    class Reduce(ReduceBase):
        def _init_reduce_dim(self, input_shape):
            self.extra_attrs["reduce_dim"] = axis
            return self.extra_attrs["reduce_dim"]

    return Reduce


def make_concat_type(arity, axis) -> Type[Concat]:
    class ConcatVariant(Concat):
        in_dtypes = [tuple(i for _ in range(arity)) for i in DTYPE_ALL]

        def __init__(self):
            super().__init__(arity)

        def _init_concat_dim(self, input_shape):
            self.extra_attrs["concat_dim"] = axis
            return self.extra_attrs["concat_dim"]

        def _init_concat_axis(self, input_shapes: List[AbsTensor]) -> int:
            self.extra_attrs["axis"] = axis
            return self.extra_attrs["axis"]

    return ConcatVariant


ATTR_FREE_RULES = [
    ElementWiseUnaryOp,
    BcastBinaryOp,
    Where,
    *[make_reduce_type(i) for i in range(__MAX_RANK__)],
    Tril,
    Triu,
    *[
        make_concat_type(arity, axis)
        for arity in range(1, Concat.MAX_ARITY + 1)
        for axis in range(__MAX_RANK__)
    ],
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
