# Utilities for using and be used by autoinf.
from dataclasses import dataclass
from functools import partialmethod
from os import PathLike
from typing import Dict, List, Tuple, Type

from autoinf.instrument.categorize import gen_inst_with_records
from autoinf.instrument.op import OpInstance

from nnsmith.abstract.dtype import DTYPE_GEN_ALL, DType
from nnsmith.abstract.op import *
from nnsmith.abstract.op import __MAX_RANK__
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
        in_dtypes = [tuple(i for _ in range(arity)) for i in DTYPE_GEN_ALL]

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


@mark_abstract("autoinf")
class AutoInfOpBase(AbsOpBase):
    @property
    def attr_names(self):
        return self.inst.A

    def __init__(self, inst: OpInstance, attrs):
        # super from self
        self.extra_attrs = {}
        self.attrs = attrs
        self.inst = inst
        self.inp_ranks = [tuple(x.rank for x in inst.input_tensors)]
        self.out_ranks = [tuple(x.rank for x in inst.output_tensors)]
        assert set(attrs.keys()) == set(inst.A), f"{list(attrs.keys())} != {inst.A}"

    def n_input(self) -> int:
        return len(self.inst.input_tensors)

    def n_output(self) -> int:
        return len(self.inst.output_tensors)

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        raise InternalError("AutoInf type_forward not implemented. Don't try it now!")

    def requires(self, input_shapes: List[AbsTensor]):
        raise InternalError("AutoInf requires not implemented. Don't try it now!")

    def deduct_inp_ranks_and_dtype(
        self: AbsOpBase, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        # check shape & type consistency
        assert len(out_abs_tensor) == len(self.out_dtypes[0])
        for i, o_aten in enumerate(out_abs_tensor):
            SanityCheck.eq(o_aten.dtype, self.out_dtypes[0][i])
            SanityCheck.eq(o_aten.ndims, self.out_ranks[0][i])

        return [
            tuple(tensor.rank, DType.from_str(tensor.dtype))
            for tensor in self.inst.input_tensors
        ]

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

    def __str__(self):
        return f"{self.inst.name}[{','.join([f'{DType.from_str(t.dtype).short()}{t.rank}' for t in self.inst.input_tensors])}]"


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

    new_type = type(
        f"__{inst.name.replace('.', '_')}_{'_'.join([f'{DType.from_str(t.dtype).short()}{t.rank}' for t in inst.input_tensors])}",
        (AutoInfOpBase,),
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
            "__module__": __name__,
        },
    )

    globals()[new_type.__name__] = new_type

    return new_type


@dataclass
class OpRecordFinder:
    producer: Dict[AbsTensor, List[AbsOpBase]]
    consumer: Dict[AbsTensor, List[AbsOpBase]]
    op2record: Dict[
        AbsOpBase, List[Tuple[Tuple[AbsTensor], Tuple[AbsTensor], Dict[str, int]]]
    ]


def make_record_finder(
    gen_inst_records: List[Tuple[OpInstance, List[Tuple[Dict]]]] = None,
    path: PathLike = None,
    max_elem_per_tensor=2**16,
):
    if gen_inst_records is None:
        assert path is not None, "Either gen_inst_records or path must be provided."
        gen_inst_records = gen_inst_with_records(data_dir=path, keep_int_value=False)

    producer = {}
    consumer = {}
    inst2record = {}

    total_rec = 0
    skipped_elem = 0
    skipped_err = 0
    skipped_blacklist = 0

    for inst, records in gen_inst_records:
        total_rec += len(records)

        if inst.name in [
            "torch.Tensor.uniform_",
            "torch.Tensor.normal_",
            "torch.Tensor.exponential_",
            "torch.Tensor.poisson_",
            "torch.Tensor.geometric_",
            "torch.Tensor.log_normal_",
            "torch.Tensor.cauchy_",
            "torch.Tensor.logistic_",
            "torch.Tensor.requires_grad_",
            "torch.nn.functional.dropout",
            "torch.nn.functional.dropout2d",
            "torch.nn.functional.dropout3d",
        ]:  # black list
            AUTOINF_LOG.error(f"Blacklist operator {inst.name} found!")
            skipped_blacklist += len(records)
            continue

        for record in records:
            try:
                input_abs_tensor = [
                    AbsTensor(shape, DType.from_str(inften.dtype))
                    for inften, shape in zip(
                        inst.input_tensors, inst.concrete_input_shapes(record[0])
                    )
                ]
                if any([x.nelement() > max_elem_per_tensor for x in input_abs_tensor]):
                    AUTOINF_LOG.debug(
                        f"Skip {inst.name} <- {input_abs_tensor} for over {max_elem_per_tensor} elements."
                    )
                    skipped_elem += 1
                    continue
            except:
                AUTOINF_LOG.error(f"{inst.name}: bad subst. {inst.I} -> {record}")
                skipped_err += 1
                continue

            try:
                output_abs_tensor = [
                    AbsTensor(shape, DType.from_str(inften.dtype))
                    for inften, shape in zip(
                        inst.output_tensors, inst.concrete_output_shapes(record[1])
                    )
                ]
                if any([x.nelement() > max_elem_per_tensor for x in output_abs_tensor]):
                    AUTOINF_LOG.debug(
                        f"Skip {inst.name} -> {output_abs_tensor} for over {max_elem_per_tensor} elements."
                    )
                    skipped_elem += 1
                    continue
            except:
                AUTOINF_LOG.error(f"{inst.name}: bad subst. {inst.O} -> {record}")
                skipped_err += 1
                continue

            for iten in input_abs_tensor:
                prod_list = producer.setdefault(iten, [])
                if inst not in prod_list:
                    prod_list.append(inst)

            for oten in output_abs_tensor:
                cons_list = consumer.setdefault(oten, [])
                if inst not in cons_list:
                    cons_list.append(inst)

            inst2record.setdefault(inst, []).append(
                (
                    tuple(input_abs_tensor),
                    tuple(output_abs_tensor),
                    {k: record[0][k] for k in inst.A},
                )
            )

    skipped_rec = skipped_elem + skipped_err + skipped_blacklist
    final_rec = total_rec - skipped_rec
    AUTOINF_LOG.info(f"Got {final_rec} records of {len(inst2record)} OpInstance")
    AUTOINF_LOG.info(f"Filtered {skipped_rec} records from {total_rec} initial set.")
    AUTOINF_LOG.info(
        f"~ {skipped_elem}: over {max_elem_per_tensor} elem.  ~ {skipped_err}: bad subst.  ~ {skipped_blacklist}: blacklisted."
    )

    return OpRecordFinder(producer, consumer, inst2record)
