import json
import os
import pickle
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from typing import Any, Dict, List, Set, Tuple, Type

import networkx as nx
import numpy as np
from multipledispatch import dispatch

from nnsmith.abstract.op import AbsOpBase, Constant, Input
from nnsmith.abstract.tensor import AbsTensor
from nnsmith.error import SanityCheck


def framework_operator_impl(
    framework_realizable_ops: List[Type[AbsOpBase]],
    all_framework_ops: List[Type[AbsOpBase]],
    op_type: AbsOpBase,
    *args,
    **kwargs,
):
    """When implementing `forward_fn` for an operator class, add this operator into all_framework_ops list.

    Usage:
        In `forward.py`, define `operator_impl = partial(framework_operator_impl, FW_REALIZABLE_OPS, ALL_FM_OPS)`.
        Then add `@operator_impl(OpClass)` when implementing `forward_fn` for `OpClass`.

    Args:
        framework_realizable_ops (List[Type[AbsOpBase]]): all realizable ops in the framework. Usually it can be obtained by FULL_OPERATOR_SETS["core"].union(FULL_OPERATOR_SETS["framework_name"])
        all_framework_ops (List[Type[AbsOpBase]]): list of operator classes that are implemented `forward_fn` in the framework.
        op_type (AbsOpBase): operator class
    """
    SanityCheck.true(
        issubclass(op_type, AbsOpBase),
        f"Decorator operator_impl takes AbsOpBase subclass, but got {op_type}",
    )
    if op_type is not Constant:  # Constant comes from placeholder.
        dispatchables = [
            rtype for rtype in framework_realizable_ops if issubclass(rtype, op_type)
        ]
        for rtype in dispatchables:
            all_framework_ops.append(rtype)

        SanityCheck.true(
            len(dispatchables) != 0,
            f"Decorator operator_impl only take types decorated by `mark_realize`, but got {op_type}",
        )
    return dispatch(op_type, *args, **kwargs)


Instruction: Tuple[AbsOpBase, List[int], List[int]] = namedtuple(
    "Instruction", ["op", "inputs", "outputs"]
)


@dataclass
class Schedule:
    """Minimal information for constructing a graph."""

    instructions: List[Instruction]
    input_keys: List[int]
    leaf_keys: List[int]
    key2type: Dict[int, AbsTensor]

    @staticmethod
    def init(graph: nx.MultiDiGraph, key2type: Dict[int, AbsTensor]) -> "Schedule":
        # The input graph should be a concretized graph.
        instructions: List[Instruction] = []
        input_keys = []
        user_keys = set()

        # freeze node with static attributes in label;
        for node_id in nx.topological_sort(graph):
            node = graph.nodes[node_id]
            op = node["op"]

            if isinstance(op, Input):
                input_keys.append(node["otensor_idx"][0])

            for used_idx in node["itensor_idx"]:
                user_keys.add(used_idx)

            # TODO(@ganler): Better name than "otensor_idx"
            # TODO(@ganler): Add refcnt or last ref mechanism to save memory
            instructions.append(
                Instruction(
                    op=op,
                    inputs=node["itensor_idx"],
                    outputs=node["otensor_idx"],
                )
            )

        # simplify the statements above
        leaf_keys = [key for key in key2type if key not in user_keys]

        return Schedule(instructions, input_keys, leaf_keys, key2type)


class Oracle:
    def __init__(
        self,
        input: Dict[str, np.ndarray],
        output: Dict[str, np.ndarray],
        provider: str = "unknown",
    ) -> None:
        self.input: Dict[str, np.ndarray] = input
        self.output: Dict[str, np.ndarray] = output
        self._provider = provider

    def __repr__(self) -> str:
        return f"input={self.input}, output={self.output}"

    @staticmethod
    def name() -> str:
        return "oracle.pkl"

    @property
    def provider(self) -> str:
        return self._provider

    def dump(self, path: PathLike) -> None:
        with open(path, "wb") as f:
            to_dump = {
                "input": self.input,
                "output": self.output,
                "provider": self.provider,
            }
            pickle.dump(to_dump, f)

    @staticmethod
    def load(path: PathLike) -> "Oracle":
        with open(path, "rb") as f:
            to_load = pickle.load(f)
            return Oracle(to_load["input"], to_load["output"], to_load["provider"])


class Model(ABC):
    @property
    @abstractmethod
    def input_like(self) -> Dict[str, AbsTensor]:
        pass

    @property
    @abstractmethod
    def output_like(self) -> Dict[str, AbsTensor]:
        pass

    @staticmethod
    @abstractmethod
    def from_schedule(schedule: Schedule, **kwargs) -> "Model":
        pass

    @staticmethod
    @abstractmethod
    def load(path: PathLike) -> "Model":
        pass

    @abstractmethod
    def dump(self, path: PathLike) -> None:
        pass

    @property
    @abstractmethod
    def native_model(self) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def name_suffix() -> str:
        """Suffix of the model file.

        Returns:
            str: Model suffix such as ".onnx".

        Note:
            Model as a folder can be "".
        """
        pass

    @abstractmethod
    def refine_weights(self) -> None:
        pass

    @staticmethod
    @abstractmethod
    def operators() -> List[Type[AbsOpBase]]:
        pass

    @staticmethod
    def name_prefix() -> str:
        return "model"

    @staticmethod
    def operators() -> List[Type[AbsOpBase]]:
        pass

    @staticmethod
    def add_seed_setter() -> None:
        pass

    @staticmethod
    def init(name) -> Type["Model"]:
        if name == "torch":
            from nnsmith.materialize.torch import TorchModel

            return TorchModel
        elif name == "onnx":
            from nnsmith.materialize.onnx import ONNXModel

            return ONNXModel
        elif name == "tensorflow":
            from nnsmith.materialize.tensorflow import TFModel

            return TFModel
        else:
            raise ValueError(f"Unknown or unsupported model type: {name}")


class TestCase:
    def __init__(self, model: Model, oracle: Oracle):
        self.oracle = oracle
        self.model = model

    @staticmethod
    def load(
        model_type: Type[Model], root_folder: str, allow_no_oracle=False
    ) -> "TestCase":
        model_path = os.path.join(
            root_folder, model_type.name_prefix() + model_type.name_suffix()
        )
        model = model_type.load(model_path)

        assert allow_no_oracle or os.path.exists(
            os.path.join(root_folder, Oracle.name())
        ), "Oracle is not found or auto-generated when allow_no_oracle is True."

        oracle = None
        if os.path.exists(os.path.join(root_folder, Oracle.name())):
            oracle_path = os.path.join(root_folder, Oracle.name())
            oracle = Oracle.load(oracle_path)

        return TestCase(model, oracle)

    def dump(self, root_folder: str):
        self.model.dump(
            os.path.join(
                root_folder, self.model.name_prefix() + self.model.name_suffix()
            )
        )
        self.oracle.dump(os.path.join(root_folder, Oracle.name()))


class Symptom(Enum):
    EXCEPTION = "exception"
    SEGFAULT = "segfault"
    INCONSISTENCY = "inconsistency"
    WORSE_PERF = "worse_perf"


class Stage(Enum):
    COMPILATION = "compilation"
    EXECUTION = "execution"
    VERIFICATION = "verification"


class BugReport(ABC):
    def __init__(
        self,
        testcase: TestCase,
        symptom: Symptom,
        stage: Stage,
        system: str,
        version: str = None,
        trigger_hash: str = None,
        log: str = None,
    ):
        self.testcase = testcase
        self.symptom = symptom
        self.system = system
        self.stage = stage
        self.version = version
        self.trigger_hash = trigger_hash
        self.log = log

    @property
    @classmethod
    def error_msg_name(cls):
        return "err.log"

    def __repr__(self) -> str:
        return f"{self.system} {self.symptom.value} in {self.stage.value}\n{self.log}"

    def dump(self, root_folder: str):
        # create folder if not exists
        if not os.path.exists(root_folder):
            os.makedirs(root_folder)

        # model*
        # oracle.pkl
        self.testcase.dump(root_folder)
        # err.log
        with open(os.path.join(root_folder, self.error_msg_name), "w") as f:
            f.write(self.log)
        # meta.json
        with open(os.path.join(root_folder, "meta.json"), "w") as f:
            json.dump(
                {
                    "system": self.system,
                    "symptom": self.symptom.value,
                    "stage": self.stage.value,
                    "version": self.version,
                    "trigger_hash": self.trigger_hash,
                },
                f,
            )

    @staticmethod
    def load(model_type, root_folder: str, allow_partial=False) -> "BugReport":

        symptom = None
        stage = None
        version = None
        trigger_hash = None
        system = None

        assert allow_partial or os.path.exists(
            os.path.join(root_folder, "meta.json")
        ), "meta.json must exist or allow_partial is True where the oracle will be automatically generated"

        if os.path.exists(os.path.join(root_folder, "meta.json")):
            with open(os.path.join(root_folder, "meta.json"), "r") as f:
                meta = json.load(f)

            system = meta["system"]
            symptom = Symptom(meta["symptom"])
            stage = Stage(meta["stage"])
            version = meta["version"]
            trigger_hash = meta["trigger_hash"]

        testcase = TestCase.load(model_type, root_folder, allow_partial)

        log = None
        if os.path.exists(os.path.join(root_folder, BugReport.error_msg_name)):
            with open(os.path.join(root_folder, BugReport.error_msg_name), "r") as f:
                log = f.read()

        return BugReport(testcase, symptom, stage, system, version, trigger_hash, log)