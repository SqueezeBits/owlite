import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from onnx.defs import OpSchema

from .onnx_op_schemas import get_core_operator_schemas

NumericValue = Union[int, float, bool, np.ndarray]


@dataclass
class FormalONNXParameter:
    """Structure wrapping properties defined in the ONNX op schema required for ONNX transformations"""

    name: str
    is_optional: bool
    is_variadic: bool
    is_homogeneous: bool
    is_differentiable: bool
    type_constraints: Sequence[np.dtype]


class ONNXOp:
    """Class representing each ONNX op allowing convenient access to its schema properties"""

    schemas: dict[str, OpSchema] = get_core_operator_schemas()

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"{self.name}"

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def is_valid(self) -> bool:
        """Checks if the op exists in schemas"""
        return self.name in ONNXOp.schemas

    @property
    def schema(self) -> OpSchema:
        """The full schema structure of the op

        Returns:
            list[tuple[int, list[tuple[str, OpSchema, list[OpSchema]]]]]: the full schema structure
        """
        return ONNXOp.schemas[self.name]

    @property
    def type_constraints(self) -> dict[str, list[str]]:
        """The dictionary that maps type parameter string to its allowed type strings

        Returns:
            dict[str, list[str]]: _description_
        """
        return {
            type_constraint.type_param_str: type_constraint.allowed_type_strs
            for type_constraint in self.schema.type_constraints
        }

    def i(self, idx: int = 0) -> "FormalONNXParameter":
        """The formal ONNX paramter of the input at given index.

        Args:
            idx (int, optional): the input index. Defaults to 0.

        Returns:
            FormalONNXParameter: the formal ONNX paramter of the input.
        """
        return self._get_formal_parameter(self.schema.inputs, idx)

    def o(self, idx: int = 0) -> "FormalONNXParameter":
        """The formal ONNX paramter of the output at given index.

        Args:
            idx (int, optional): the output index. Defaults to 0.

        Returns:
            FormalONNXParameter: the formal ONNX paramter of the output.
        """
        return self._get_formal_parameter(self.schema.outputs, idx)

    def _get_formal_parameter(self, params: list, idx: int = 0) -> FormalONNXParameter:
        is_last_parameter_variadic = params[-1].option == OpSchema.FormalParameterOption.Variadic
        if not (-len(params) <= idx < len(params) or is_last_parameter_variadic):
            raise IndexError(f"{self.name}: index out of range: {idx}")
        if is_last_parameter_variadic:
            param_idx = min(idx, len(params) - 1)
            offset = idx - param_idx
            param = params[param_idx]
            param_name = f"{param.name}_{offset}"
        else:
            param = params[idx]
            param_name = param.name
        return FormalONNXParameter(
            name=param_name,
            is_optional=OpSchema.FormalParameterOption.Optional == param.option,
            is_variadic=OpSchema.FormalParameterOption.Variadic == param.option,
            is_homogeneous=param.is_homogeneous,
            is_differentiable=OpSchema.DifferentiationCategory.Differentiable == param.differentiation_category,
            type_constraints=convert_to_np_dtypes(self.type_constraints.get(param.type_str, param.type_str)),
        )


def convert_to_np_dtypes(wrapped_type_strs: list[str]) -> list[np.dtype]:
    """Converts type strings from an op schema to numpy data type

    Args:
        wrapped_type_strs (list[str]): the op schema type string

    Returns:
        list[np.dtype]: the converted numpy data type.
    """
    return [
        dtype for dtype in map(try_convert_to_np_dtype, map(unwrap_type_str, wrapped_type_strs)) if dtype is not None
    ]


def unwrap_type_str(type_str: str) -> str:
    """Unwraps a type string from an op schema if possible

    Args:
        type_str (str): an op schema type string

    Returns:
        str: the string containing only type name if the unwrapping was successful, the input type_str itself otherwise.
    """
    match = re.search(r"tensor\((.*?)\)", type_str)
    if match:
        # Return the extracted string
        return match.group(1)
    # Return the input itself if no match is found
    return type_str


def try_convert_to_np_dtype(type_str: str) -> Optional[np.dtype]:
    """Converts the type name in string into numpy data type if possible.

    Args:
        type_str (str): a string containing type name

    Returns:
        Optional[np.dtype]: a numpy.dtype instance if the conversion was successful, None otherwise.
    """
    if type_str == "float":
        type_str = "float32"
    try:
        return np.dtype(type_str)
    except TypeError:
        pass
    return None
