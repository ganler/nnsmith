# Utilities for using and be used by autoinf.
from dataclasses import dataclass
from functools import partialmethod
from typing import Dict, FrozenSet, List, Tuple, Type

from autoinf.instrument.op import OpInstance

from nnsmith.abstract.dtype import DType
from nnsmith.abstract.op import *
from nnsmith.abstract.tensor import AbsTensor
from nnsmith.error import InternalError, SanityCheck
from nnsmith.logging import AUTOINF_LOG


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


class AutoInfOpBase:
    @classmethod
    def attr_names(cls):
        return cls._attr_names


def make_autoinf_op_type(inst: OpInstance):
    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        raise InternalError("AutoInf type_forward not implemented. Don't try it now!")

    def requires(self, input_shapes: List[AbsTensor]):
        raise InternalError("AutoInf requires not implemented. Don't try it now!")

    def deduct_inp_ranks_and_dtype(
        self: AbsOpBase, inst: OpInstance, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        # check shape & type consistency
        assert len(out_abs_tensor) == len(self.out_dtypes[0])
        for i, o_aten in enumerate(out_abs_tensor):
            SanityCheck.eq(o_aten.dtype, self.out_dtypes[0][i])
            SanityCheck.eq(o_aten.ndims, self.out_ranks[0][i])

        return [
            tuple(tensor.rank, DType.from_str(tensor.dtype))
            for tensor in inst.input_tensors
        ]

    def init_from_inst(self: AbsOpBase, inst: OpInstance, attr):
        # super from self
        super(type(self), self).__init__()
        self.inp_ranks = [tuple(x.rank for x in inst.input_tensors)]
        self.out_ranks = [tuple(x.rank for x in inst.output_tensors)]
        assert set(attr.keys()) == set(inst.A), f"{list(attr.keys())} != {inst.A}"
        self.attrs = attr

    def make_substition(
        self: AbsOpBase, inst: OpInstance, input_shapes: List[AbsTensor]
    ):
        symbol_subst = {}
        # input shape subst
        for inf_ten, smt_ten in zip(inst.input_tensors, input_shapes):
            for name, val in zip(inf_ten.shape, smt_ten.shape):
                symbol_subst[name] = val
        # attr subst | update self.attrs
        symbol_subst.update(self.attrs)
        return symbol_subst

    def concretize(inst: OpInstance, attr_map):
        # attr_map: {attr_name: attr_value}
        callable = inst.materialize(eval(inst.name), attr_map)
        return callable

    new_type = type(
        f"AutoInfOp::{{{inst.name}}}_{'_'.join([f'{DType.from_str(t.dtype).short()}{t.rank}' for t in inst.input_tensors])}",
        (
            AbsOpBase,
            AutoInfOpBase,
        ),
        {
            "in_dtypes": [tuple(DType.from_str(x.dtype) for x in inst.input_tensors)],
            "out_dtypes": [tuple(DType.from_str(x.dtype) for x in inst.output_tensors)],
            "_attr_names": inst.A,
            "__init__": partialmethod(init_from_inst, inst),
            "make_substition": partialmethod(make_substition, inst),
            "type_transfer": partialmethod(type_transfer),
            "requires": partialmethod(requires),
            "deduct_inp_ranks_and_dtype": partialmethod(
                deduct_inp_ranks_and_dtype, inst
            ),
            "inst": inst,
        },
    )

    return new_type


@dataclass
class OpRecordFinder:
    producer: Dict[FrozenSet[AbsTensor], List[AbsOpBase]]
    consumer: Dict[FrozenSet[AbsTensor], List[AbsOpBase]]


def make_op_record_finder(gen_inst_records: List[Tuple[OpInstance, List[Tuple[Dict]]]]):
    producer = {}
    consumer = {}
    for inst, records in gen_inst_records:
        op_type = make_autoinf_op_type(inst)
        for record in records:
            try:
                input_abs_tensor = [
                    AbsTensor(shape, DType.from_str(inften.dtype))
                    for inften, shape in zip(
                        inst.input_tensors, inst.concrete_input_shapes(record[0])
                    )
                ]
            except:
                AUTOINF_LOG.error(f"{inst.name}: bad subst. {inst.I} -> {record}")
                continue

            try:
                output_abs_tensor = [
                    AbsTensor(shape, DType.from_str(inften.dtype))
                    for inften, shape in zip(
                        inst.output_tensors, inst.concrete_output_shapes(record[1])
                    )
                ]
            except:
                AUTOINF_LOG.error(f"{inst.name}: bad subst. {inst.O} -> {record}")
                continue

            producer.setdefault(frozenset(input_abs_tensor), []).append(op_type)
            consumer.setdefault(frozenset(output_abs_tensor), []).append(op_type)

    return OpRecordFinder(producer, consumer)
