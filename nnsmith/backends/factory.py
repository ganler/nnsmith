from abc import ABC, abstractmethod
from typing import List, Union, Dict, Tuple
from collections import namedtuple

import onnx
import numpy as np

ShapeType = namedtuple('ShapeType', ['shape', 'dtype'])


class BackendFactory(ABC):
    def __init__(self, device='cpu', optmax=True):
        self.name = 'tvm'
        super().__init__()
        self.device = device
        self.optmax = optmax
        assert self.name, 'Need to add`self.name = "my_backend"` during extension'

    @abstractmethod
    def mk_backend(self, model: Union[onnx.ModelProto, str], **kwargs) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    @staticmethod
    def get_onnx_proto(model: Union[onnx.ModelProto, str]) -> onnx.ModelProto:
        if isinstance(model, str):
            onnx_model = onnx.load(model)
        else:
            assert isinstance(model, onnx.ModelProto)
            onnx_model = model
        return onnx_model

    @classmethod
    def _coverage_install(cls):
        raise NotImplementedError("Coverage support not implemented.")

    @classmethod
    def coverage_install(cls):
        try:
            return cls._coverage_install()
        except Exception as e:
            print(f'Coverage support not implemented: {e}')
            print(f'Falling back to no coverage support.')

            class FallbackCoverage:
                @staticmethod
                def get_now():
                    return 0

                @staticmethod
                def get_total():
                    return 0

            return FallbackCoverage()

    @staticmethod
    def dtype_str(id: int) -> str:
        """See https://deeplearning4j.org/api/latest/onnx/Onnx.TensorProto.DataType.html
        """
        if id == 1:
            return 'float32'
        elif id == 2:
            return 'uint8'
        elif id == 3:
            return 'int8'
        elif id == 4:
            return 'uint16'
        elif id == 5:
            return 'int16'
        elif id == 6:
            return 'int32'
        elif id == 7:
            return 'int64'
        elif id == 8:
            return 'string'
        elif id == 9:
            return 'bool'
        elif id == 10:
            return 'float16'
        elif id == 11:
            return 'double'
        elif id == 12:
            return 'uint32'
        elif id == 13:
            return 'uint64'
        elif id == 14:
            return 'complex64'
        elif id == 15:
            return 'complex128'
        elif id == 16:
            return 'bfloat16'
        else:
            raise ValueError(
                f'Unknown dtype id: {id}. See https://deeplearning4j.org/api/latest/onnx/Onnx.TensorProto.DataType.html')

    @staticmethod
    def analyze_onnx_io(model: onnx.ModelProto) -> Tuple[Dict[str, ShapeType], List[str]]:
        """Analyze the input and output shapes of an ONNX model.

        Args:
            model (onnx.ModelProto): The model to be analyzed.

        Returns:
            Tuple[Dict[str, ShapeType], List[str]]: Input specifications (name -> {shape, dtype}) and output names.
        """
        inp_analysis_ret = {}
        out_analysis_names = [node.name for node in model.graph.output]

        # Note that there are 2 kinds of "inputs":
        # 1. The inputs provided by the user (e.g., images);
        # 2. The inputs provided by the model (e.g., the weights).
        # We only consider the first kind of inputs.
        weight_names = [node.name for node in model.graph.initializer]

        # Analyze the input shapes
        # Expected information:
        #   For each input:
        #     1. name
        #     2. shape (Note: `-1` stands for unknown dimension)
        #     3. data type
        # iterate through inputs of the graph
        for input_node in model.graph.input:
            if input_node.name in weight_names:
                continue
            # get type of input tensor
            tensor_type = input_node.type.tensor_type

            shape = []
            dtype = BackendFactory.dtype_str(tensor_type.elem_type)

            # check if it has a shape:
            if (tensor_type.HasField("shape")):
                # iterate through dimensions of the shape:
                for d in tensor_type.shape.dim:
                    # the dimension may have a definite (integer) value or a symbolic identifier or neither:
                    if (d.HasField("dim_value")):
                        shape.append(d.dim_value)  # known dimension
                    elif (d.HasField("dim_param")):
                        # unknown dimension with symbolic name
                        shape.append(-1)
                    else:
                        shape.append(-1)  # unknown dimension with no name
            else:
                raise ValueError(
                    "Input node {} has no shape".format(input_node.name))

            inp_analysis_ret[input_node.name] = ShapeType(shape, dtype)

        return inp_analysis_ret, out_analysis_names