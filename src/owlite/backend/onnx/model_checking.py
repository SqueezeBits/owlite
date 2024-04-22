import os
from collections import OrderedDict

import numpy as np
import onnx
import onnx.checker
import onnxruntime as rt

from ...owlite_core.logger import log
from ..utils import compare_nested_outputs

Tensors = dict[str, np.ndarray]
TensorShape = list[int]
TensorShapes = dict[str | None, TensorShape]


# This function has been modified from onnxsim.model_checking.compare
# (See https://github.com/daquexian/onnx-simplifier/blob/master/onnxsim/model_checking.py)
# pylint: disable=too-many-locals, too-many-statements
def compare(
    model_opt: str | onnx.ModelProto,
    model_ori: str | onnx.ModelProto,
    n_times: int = 5,
    input_shapes: TensorShapes | None = None,
    input_data: Tensors | None = None,
    custom_lib: str | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    equal_nan: bool = False,
) -> bool:
    """Compare two ONNX models.

    Args:
        model_opt (str | onnx.ModelProto): The simplified ONNX model.
        model_ori (str | onnx.ModelProto): The original ONNX model.
        n_times (int, optional): Generate n random inputs. Defaults to 5.
        input_shapes (TensorShapes | None, optional): Shapes of generated random inputs. Defaults to None.
        input_data (Tensors | None, optional): User-given data instead of random generated data. Defaults to None.
        custom_lib (str | None, optional): ONNX Runtime custom lib for custom ops. Defaults to None.
        rtol (float | None, optional): The relative tolerance parameter. Defaults to None.
        atol (float | None, optional): The absolute tolerance parameter. Defaults to None.
        equal_nan (bool, optional): Whether to compare NaN's as equal.
            If True, NaN's in `a` will be considered equal to NaN's in `b` in the output array. Defaults to False.

    Returns:
        bool: True if the models are equal within the specified tolerance, False otherwise.
    """

    def get_shape_from_value_info_proto(v: onnx.ValueInfoProto) -> list[int]:
        return [dim.dim_value for dim in v.type.tensor_type.shape.dim]

    def get_value_info_all(m: onnx.ModelProto, name: str) -> onnx.ValueInfoProto | None:
        for v in m.graph.value_info:
            if v.name == name:
                return v

        for v in m.graph.input:
            if v.name == name:
                return v

        for v in m.graph.output:
            if v.name == name:
                return v

        return None

    def get_shape(m: onnx.ModelProto, name: str) -> TensorShape:
        """Get the shape of an input or output tensor from an ONNX model.

        Note: This method relies on onnx shape inference, which is not reliable.
        So only use it on input or output tensors.
        """
        v = get_value_info_all(m, name)
        if v is not None:
            return get_shape_from_value_info_proto(v)
        raise RuntimeError(f'Cannot get shape of "{name}"')

    def get_elem_type(m: onnx.ModelProto, name: str) -> int | None:
        v = get_value_info_all(m, name)
        if v is not None:
            return v.type.tensor_type.elem_type
        return None

    def get_np_type_from_elem_type(elem_type: int) -> int:
        sizes = (
            None,
            np.float32,
            np.uint8,
            np.int8,
            np.uint16,
            np.int16,
            np.int32,
            np.int64,
            str,
            bool,
            np.float16,
            np.double,
            np.uint32,
            np.uint64,
            np.complex64,
            np.complex128,
            np.float16,
        )
        assert len(sizes) == 17
        size = sizes[elem_type]
        assert size is not None
        return size

    def get_input_names(model: onnx.ModelProto) -> list[str]:
        input_names = list({ipt.name for ipt in model.graph.input} - {x.name for x in model.graph.initializer})
        return input_names

    def generate_rand_input(model: str | onnx.ModelProto, input_shapes: TensorShapes | None = None):
        if input_shapes is None:
            input_shapes = {}
        if isinstance(model, str):
            model = onnx.load(model, load_external_data=False)
        input_names = get_input_names(model)
        full_input_shapes = {ipt: get_shape(model, ipt) for ipt in input_names}
        if None in input_shapes:
            log.debug("input_shapes:")
            log.debug(input_shapes)
            raise ValueError("None is exist in input shapes")
        full_input_shapes.update(input_shapes)  # type: ignore
        for name, shape in full_input_shapes.items():
            if any(dim <= 0 for dim in shape[1:]):
                raise RuntimeError(
                    f'The shape of input "{name}" has dynamic size, '
                    "please set an input shape manually with --test-input-shape"
                )
            if len(shape) > 0 and shape[0] <= 0:  # pylint: disable=chained-comparison
                print(
                    f'shape[0] of input "{name}" is dynamic, we assume it presents batch size and set it as 1 when '
                    "testing. If it is not wanted, please set the it manually by --test-input-shape "
                    "(see `onnxsim -h` for the details)."
                )
                shape[0] = 1

        inputs = {
            ipt: np.array(
                np.random.rand(*full_input_shapes[ipt]),
                dtype=get_np_type_from_elem_type(get_elem_type(model, ipt)),
            )
            for ipt in input_names
        }
        return inputs

    def forward(
        model: str | onnx.ModelProto,
        inputs: Tensors,
        custom_lib: str | None = None,
    ) -> dict[str, np.ndarray]:
        sess_options = rt.SessionOptions()
        if custom_lib is not None:
            if os.path.exists(custom_lib):
                sess_options.register_custom_ops_library(custom_lib)
            else:
                raise ValueError(f"No such file '{custom_lib}'")
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel(0)
        sess_options.log_severity_level = 3
        if isinstance(model, onnx.ModelProto):
            model = model.SerializeToString()
        sess = rt.InferenceSession(
            model,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        outputs = [x.name for x in sess.get_outputs()]
        run_options = rt.RunOptions()
        run_options.log_severity_level = 3
        res = OrderedDict(zip(outputs, sess.run(outputs, inputs, run_options=run_options)))
        return res

    if input_shapes is None:
        input_shapes = {}
    onnx.checker.check_model(model_opt)
    for i in range(n_times):
        print(f"Checking {i}/{n_times}...")
        if input_data is None:
            inputs = generate_rand_input(model_opt, input_shapes=input_shapes)
        else:
            inputs = input_data
        res_ori = forward(model_ori, inputs, custom_lib)
        res_opt = forward(model_opt, inputs, custom_lib)

        if not compare_nested_outputs(res_opt, res_ori, rtol=rtol, atol=atol, equal_nan=equal_nan):
            return False
    return True
