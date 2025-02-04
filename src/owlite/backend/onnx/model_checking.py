import json
import os
from dataclasses import dataclass
from itertools import chain

import numpy as np
import onnx
import onnx.checker
import onnxruntime as ort
from typing_extensions import Self

from ...core.logger import log
from ..utils import compare_pytrees, get_numpy_type, get_onnx_tensor_shape


@dataclass
class NDArrayProto:
    """Shape and dtype for creating a numpy array."""

    shape: tuple[int, ...]
    dtype: np.dtype

    def rand(self) -> np.ndarray:
        """Generate random numpy array with shape and dtype.

        Returns:
            np.ndarray: a randomly generated numpy array with `shape=self.shape` and `dtype=self.dtype`.
        """
        return np.random.rand(*self.shape).astype(self.dtype)

    @classmethod
    def extract_from(cls, value: onnx.ValueInfoProto) -> Self | None:
        """Extract numpy value info from ONNX value info proto.

        Args:
            value (onnx.ValueInfoProto): an ONNX value info proto.

        Returns:
            Self | None: the extracted numpy value info if it is numpy-compatible, None otherwise.
        """
        if (dtype := get_numpy_type(value)) is None:
            log.error(f"The ONNX input {value.name} has an element type not supported by numpy.")  # UX
            return None
        raw_shape = get_onnx_tensor_shape(value)
        if raw_shape is None or len(raw_shape) != len(shape := tuple(s for s in raw_shape if isinstance(s, int))):
            log.error(f"The ONNX input {value.name} has dynamic shape: {raw_shape}")  # UX
            return None
        return cls(shape=shape, dtype=dtype)

    def __str__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype.name})"


class NDArraySignature(dict[str, NDArrayProto]):
    """Dictionary holding named ndarray protos."""

    @classmethod
    def extract_from_inputs(cls, model: onnx.ModelProto) -> Self | None:
        """Extract numpy value info objects from the ONNX model proto's inputs.

        Args:
            model (onnx.ModelProto): an ONNX model proto.

        Returns:
            Self | None: the extracted numpy value info dictionary if all of the inputs of the ONNX model proto are
                numpy-compatible, None otherwise.
        """
        x = cls()
        for v in model.graph.input:
            if value_info := NDArrayProto.extract_from(v):
                x[v.name] = value_info
                continue
            return None
        return x

    def generate_random(self) -> dict[str, np.ndarray]:
        """Generate dictionary of random numpy arrays based on numpy value info objects it is holding.

        Returns:
            dict[str, np.ndarray]: the dictionary of random numpy arrays corresponding to the numpy value info objects.
        """
        return {name: value.rand() for name, value in self.items()}


def compare(
    first: bytes | str | onnx.ModelProto,
    second: bytes | str | onnx.ModelProto,
    *,
    n_times: int = 1,
    custom_lib: str | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    equal_nan: bool = False,
    first_name: str | None = None,
    second_name: str | None = None,
) -> bool:
    """Compare two ONNX models.

    Args:
        first (bytes | str | onnx.ModelProto): An object conveying ONNX ModelProto.
        second (bytes | str | onnx.ModelProto): Another object conveying ONNX ModelProto.
        n_times (int, optional): Number of times to compare functionality with random inputs. Defaults to 1.
        custom_lib (str | None, optional): ONNX Runtime custom lib for custom ops. Defaults to None.
        rtol (float | None, optional): The relative tolerance parameter. Defaults to None.
        atol (float | None, optional): The absolute tolerance parameter. Defaults to None.
        equal_nan (bool, optional): Whether to compare NaN's as equal.
            If True, NaN's in `a` will be considered equal to NaN's in `b` in the output array. Defaults to False.
        first_name (str, optional): the name of the first ONNX ModelProto object for debugging.
        second_name (str, optional): the name of the second ONNX ModelProto object for debugging.

    Returns:
        bool: True if the models are equal within the specified tolerance, False otherwise.
    """
    first_name = first_name or "first ONNX model proto"
    second_name = second_name or "second ONNX model proto"

    try:
        first_signature, second_signature = (
            get_ndarray_signature(name, model_proto, warn_bfloat16=i == 0)
            for i, (name, model_proto) in enumerate(
                (
                    (first_name, load_for_inspection(first)),
                    (second_name, load_for_inspection(second)),
                )
            )
        )
    except RuntimeError as e:
        log.error(e)
        return False

    if [*first_signature.values()] != [*second_signature.values()]:
        log.error(
            "The two ONNX model protos have different signature:\n"
            f"  * {first_name}: {json.dumps([str(x) for x in first_signature.values()], indent=2)}\n"
            f"  * {second_name}: {json.dumps([str(x) for x in second_signature.values()], indent=2)}\n"
        )  # UX
        return False

    for i in range(n_times):
        log.debug(f"Checking {i}/{n_times}...")
        inputs_for_first = first_signature.generate_random()
        inputs_for_second = dict(zip(second_signature.keys(), inputs_for_first.values()))
        if not compare_pytrees(
            run_inference(first, inputs_for_first, custom_lib),
            run_inference(second, inputs_for_second, custom_lib),
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        ):
            return False
    return True


def get_ndarray_signature(name: str, model_proto: onnx.ModelProto, warn_bfloat16: bool = False) -> NDArraySignature:
    """Get input ndarray signature of the ONNX model proto, optionally warning about bfloat16 data type if found.

    Args:
        name (str): the name of the ONNX model proto for warning.
        model_proto (onnx.ModelProto): the ONNX model proto.
        warn_bfloat16 (bool, optional): whether to warn about bfloat16 tensors if any found. Defaults to False.

    Raises:
        RuntimeError: if any of the input tensor has element type not supported by numpy.

    Returns:
        NDArraySignature: the ndarray signature of the ONNX model proto.
    """
    if warn_bfloat16 and has_bfloat16_tensor(model_proto):
        log.warning(
            f"The {name} has one or more bfloat16 tensors. The ONNX functionality checking might fail as numpy "
            "doesn't support bfloat16 and onnxruntime is missing bfloat16 implementations for some ONNX ops."
        )  # UX
    if (signature := NDArraySignature.extract_from_inputs(model_proto)) is None:
        raise RuntimeError(f"Failed to extract valid shapes and types from the {name} ONNX model proto")  # UX
    return signature


def load_for_inspection(model: bytes | str | onnx.ModelProto) -> onnx.ModelProto:
    """Load the ONNX model representation for inspection only.

    Args:
        model (bytes | str | onnx.ModelProto): an object representing an ONNX model proto.

    Returns:
        onnx.ModelProto: the ONNX model proto loaded possibly without external data.
    """
    if isinstance(model, onnx.ModelProto):
        return model
    if isinstance(model, str):
        return onnx.load(model, load_external_data=False)
    return onnx.load_from_string(model)


def run_inference(
    model: bytes | str | onnx.ModelProto,
    inputs: dict[str, np.ndarray],
    custom_lib: str | None = None,
) -> dict[str, np.ndarray]:
    """Run the model inference with the given inputs.

    Args:
        model (bytes | str | onnx.ModelProto): an object representing an ONNX model proto.
        inputs (dict[str, np.ndarray]): named input numpy arrays.
        custom_lib (str | None, optional): ONNX Runtime custom lib for custom ops. Defaults to None.

    Raises:
        FileNotFoundError: if the provided `custom_lib` is not found.

    Returns:
        dict[str, np.ndarray]: the named output numpy arrays.
    """
    sess_options = ort.SessionOptions()
    if custom_lib is not None:
        if os.path.exists(custom_lib):
            sess_options.register_custom_ops_library(custom_lib)
        else:
            raise FileNotFoundError(f"No such file '{custom_lib}'")
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel(0)
    sess_options.log_severity_level = 2
    sess = ort.InferenceSession(
        (model.SerializeToString() if isinstance(model, onnx.ModelProto) else model),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    run_options = ort.RunOptions()
    run_options.log_severity_level = 2
    return dict(
        zip(
            # pylint: disable-next=not-an-iterable
            (output_names := [x.name for x in sess.get_outputs()]),
            sess.run(output_names, inputs, run_options=run_options),
        )
    )


def has_bfloat16_tensor(model: onnx.ModelProto) -> bool:
    """Check if the ONNX model proto has any bfloat16 tensor.

    Args:
        model (onnx.ModelProto): an ONNX model proto.

    Returns:
        bool: True if at least one bfloat16 tensor found, False otherwise.
    """
    return any(
        chain(
            (initializer.data_type == onnx.TensorProto.BFLOAT16 for initializer in model.graph.initializer),
            (
                value_info.type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16
                for value_info in chain(model.graph.value_info, model.graph.input, model.graph.output)
            ),
        )
    )
