# pylint: disable=invalid-name, line-too-long
# ruff: noqa: E501
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property
from typing import Any, Optional

import numpy as np
import onnx
from onnx import defs
from typing_extensions import Self

from ...options.options_dict import OptionsDict
from ...options.options_mixin import OptionsMixin


class FormalParameterOption(Enum):
    """
    A statically analyzable python class for
    [OpSchema.FormalParameterOption](https://github.com/onnx/onnx/blob/v1.15.0/onnx/onnx_cpp2py_export/defs.pyi#L93-L96)
    """

    Single = 0
    Optional = auto()
    Variadic = auto()


class DifferentiationCategory(Enum):
    """
    A statically analyzable python class for
    [OpSchema.DifferentiationCategory](https://github.com/onnx/onnx/blob/v1.15.0/onnx/onnx_cpp2py_export/defs.pyi#L98-L101)
    """

    Unknown = 0
    Differentiable = auto()
    NonDifferentiable = auto()


@dataclass
class FormalParameter(OptionsMixin):
    """
    A statically analyzable python class for
    [OpSchema.FormalParameter](https://github.com/onnx/onnx/blob/v1.15.0/onnx/onnx_cpp2py_export/defs.pyi#L103-L130)
    """

    name: str
    type_str: str
    description: str
    option: FormalParameterOption
    is_homogeneous: bool
    min_arity: int
    differentiation_category: DifferentiationCategory

    @property
    def is_single(self) -> bool:
        """Equivalent to `self.option == FormalParameterOption.Single`"""
        return self.option == FormalParameterOption.Single

    @property
    def is_optional(self) -> bool:
        """Equivalent to `self.option == FormalParameterOption.Optional`"""
        return self.option == FormalParameterOption.Optional

    @property
    def is_variadic(self) -> bool:
        """Equivalent to `self.option == FormalParameterOption.Variadic`"""
        return self.option == FormalParameterOption.Variadic

    @classmethod
    def from_defs(cls, parameter: defs.OpSchema.FormalParameter) -> Self:
        """Instantiation from the original class in `onnx.defs`"""
        return cls(
            name=parameter.name,
            type_str=parameter.type_str,
            description=parameter.description,
            option=FormalParameterOption(parameter.option.value),
            is_homogeneous=parameter.is_homogeneous,
            min_arity=parameter.min_arity,
            differentiation_category=DifferentiationCategory(parameter.differentiation_category.value),
        )


@dataclass
class CompiledFormalParameter(OptionsMixin):
    """The formal parameter compiled for an input or output at a specific index"""

    name: str
    allowed_types: list[np.dtype]
    description: str
    option: FormalParameterOption
    is_homogeneous: bool
    min_arity: int
    differentiation_category: DifferentiationCategory

    @property
    def is_single(self) -> bool:
        """Equivalent to `self.option == FormalParameterOption.Single`"""
        return self.option == FormalParameterOption.Single

    @property
    def is_optional(self) -> bool:
        """Equivalent to `self.option == FormalParameterOption.Optional`"""
        return self.option == FormalParameterOption.Optional

    @property
    def is_variadic(self) -> bool:
        """Equivalent to `self.option == FormalParameterOption.Variadic`"""
        return self.option == FormalParameterOption.Variadic


@dataclass
class TypeConstraintParam(OptionsMixin):
    """
    A statically analyzable python class for
    [OpSchema.TypeConstraintParam](https://github.com/onnx/onnx/blob/v1.15.0/onnx/onnx_cpp2py_export/defs.pyi#L72-L91)
    """

    type_param_str: str
    allowed_type_strs: list[str]
    description: str

    @cached_property
    def allowed_types(self) -> list[np.dtype]:
        """The allowed types converted into np.dtype instances"""
        return convert_to_np_dtypes(self.allowed_type_strs)

    @classmethod
    def from_defs(cls, constraint: defs.OpSchema.TypeConstraintParam) -> Self:
        """Instantiation from the original class in `onnx.defs`"""
        return cls(
            type_param_str=constraint.type_param_str,
            allowed_type_strs=list(constraint.allowed_type_strs),
            description=constraint.description,
        )


class TypeConstraintParamMap(OptionsDict[str, TypeConstraintParam]):
    """
    * Key (str): a type parameter string
    * Value (TypeConstraintParam): the type constraint corresponding to the type parameter string
    """


class AttrType(Enum):
    """
    A statically analyzable python class for
    [AttrType](https://github.com/onnx/onnx/blob/v1.15.0/onnx/onnx_cpp2py_export/defs.pyi#L132-L146)
    """

    NONE = 0
    FLOAT = auto()
    INT = auto()
    STRING = auto()
    TENSOR = auto()
    GRAPH = auto()
    SPARSE_TENSOR = auto()
    TYPE_PROTO = auto()
    FLOATS = auto()
    INTS = auto()
    STRINGS = auto()
    TENSORS = auto()
    GRAPHS = auto()
    SPARSE_TENSORS = auto()
    TYPE_PROTOS = auto()


@dataclass
class AttributeProto(OptionsMixin):
    """A statically analyzable python class for `onnx.AttributeProto`"""

    name: str
    type: AttrType
    value: Any

    @classmethod  # pylint: disable-next=too-many-statements, too-many-return-statements
    def from_defs(cls, attr_proto: onnx.AttributeProto) -> Optional[Self]:
        """Instantiation from the original class in `onnx.defs`"""
        attr_type = AttrType(attr_proto.type)
        match attr_type:
            case AttrType.NONE:
                return None
            case AttrType.FLOAT:
                return cls(attr_proto.name, attr_type, attr_proto.f)
            case AttrType.INT:
                return cls(attr_proto.name, attr_type, attr_proto.i)
            case AttrType.STRING:
                return cls(attr_proto.name, attr_type, attr_proto.s.decode("UTF-8"))
            case AttrType.TENSOR:
                return cls(attr_proto.name, attr_type, attr_proto.t)
            case AttrType.GRAPH:
                return cls(attr_proto.name, attr_type, attr_proto.g)
            case AttrType.SPARSE_TENSOR:
                return cls(attr_proto.name, attr_type, attr_proto.sparse_tensor)
            case AttrType.TYPE_PROTO:
                return cls(attr_proto.name, attr_type, attr_proto.tp)
            case AttrType.FLOATS:
                return cls(attr_proto.name, attr_type, attr_proto.floats)
            case AttrType.INTS:
                return cls(attr_proto.name, attr_type, attr_proto.ints)
            case AttrType.STRINGS:
                return cls(attr_proto.name, attr_type, attr_proto.strings)
            case AttrType.TENSORS:
                return cls(attr_proto.name, attr_type, attr_proto.tensors)
            case AttrType.GRAPHS:
                return cls(attr_proto.name, attr_type, attr_proto.graphs)
            case AttrType.SPARSE_TENSORS:
                return cls(attr_proto.name, attr_type, attr_proto.sparse_tensors)
            case AttrType.TYPE_PROTOS:
                return cls(attr_proto.name, attr_type, attr_proto.type_protos)


@dataclass
class Attribute(OptionsMixin):
    """
    A statically analyzable python class for
    [OpSchema.Attribute](https://github.com/onnx/onnx/blob/v1.15.0/onnx/onnx_cpp2py_export/defs.pyi#L148-L174)
    """

    name: str
    type: AttrType
    description: str
    default_value: Optional[AttributeProto]
    required: bool

    @classmethod
    def from_defs(cls, attribute: defs.OpSchema.Attribute) -> Self:
        """Instantiation from the original class in `onnx.defs`"""
        return cls(
            name=attribute.name,
            type=AttrType(attribute.type),
            description=attribute.description,
            default_value=AttributeProto.from_defs(attribute.default_value),
            required=attribute.required,
        )


class AttributeMap(OptionsDict[str, Attribute]):
    """
    * Key (str): the name of an attribute
    * Value (Attribute): the attribute
    """


@dataclass  # pylint: disable-next=too-many-instance-attributes
class OpSchema(OptionsMixin):
    """
    A statically analyzable python class for
    [OpSchema](https://github.com/onnx/onnx/blob/v1.15.0/onnx/onnx_cpp2py_export/defs.pyi#L10-L70)
    """

    name: str
    domain: str
    since_version: int
    doc: str
    type_constraints: TypeConstraintParamMap
    inputs: list[FormalParameter]
    outputs: list[FormalParameter]
    attributes: AttributeMap
    min_input: int
    max_input: int
    min_output: int
    max_output: int

    @classmethod
    def from_defs(cls, schema: defs.OpSchema) -> Self:
        """Instantiation from the original class in `onnx.defs`"""
        return cls(
            name=schema.name,
            domain=schema.domain,
            since_version=schema.since_version,
            doc=schema.doc,
            type_constraints=TypeConstraintParamMap(
                {
                    type_constraint.type_param_str: TypeConstraintParam.from_defs(type_constraint)
                    for type_constraint in schema.type_constraints
                }
            ),
            inputs=[FormalParameter.from_defs(p) for p in schema.inputs],
            outputs=[FormalParameter.from_defs(p) for p in schema.outputs],
            attributes=AttributeMap({name: Attribute.from_defs(attr) for name, attr in schema.attributes.items()}),
            min_input=schema.min_input,
            max_input=schema.max_input,
            min_output=schema.min_output,
            max_output=schema.max_output,
        )

    def i(self, idx: int = 0) -> CompiledFormalParameter:
        """The formal parameter of the input at given index.

        Args:
            idx (int, optional): the input index. Defaults to 0.

        Returns:
            CompiledFormalParameter: the formal ONNX parameter of the input.
        """
        # Ideally, this would've been `return self.inputs[idx]`, but the reality is not so simple.
        return _get_formal_parameter(idx, self.inputs, self.type_constraints)

    def o(self, idx: int = 0) -> CompiledFormalParameter:
        """The formal parameter of the output at given index.

        Args:
            idx (int, optional): the output index. Defaults to 0.

        Returns:
            CompiledFormalParameter: the formal ONNX parameter of the output.
        """
        # Ideally, this would've been `return self.outputs[idx]`, but the reality is not so simple.
        return _get_formal_parameter(idx, self.outputs, self.type_constraints)


def _get_formal_parameter(
    idx: int,
    params: list[FormalParameter],
    type_constraints: TypeConstraintParamMap,
) -> CompiledFormalParameter:
    is_last_parameter_variadic = params[-1].is_variadic
    if not (-len(params) <= idx < len(params) or is_last_parameter_variadic):
        raise IndexError(f"input or output index out of range: {idx}")
    if is_last_parameter_variadic:
        param_idx = min(idx, len(params) - 1)
        offset = idx - param_idx
        param = params[param_idx]
        name = f"{param.name}_{offset}"
    else:
        param = params[idx]
        name = param.name
    return CompiledFormalParameter(
        name=name,
        allowed_types=_get_type_contraints(param.type_str, type_constraints).allowed_types,
        description=param.description,
        option=param.option,
        is_homogeneous=param.is_homogeneous,
        min_arity=param.min_arity,
        differentiation_category=param.differentiation_category,
    )


def _get_type_contraints(type_str: str, type_constraints: TypeConstraintParamMap) -> TypeConstraintParam:
    if type_str in type_constraints:
        return type_constraints[type_str]
    return TypeConstraintParam(type_str, [type_str], description="")


def convert_to_np_dtypes(wrapped_type_strs: list[str]) -> list[np.dtype]:
    """Converts type strings from an op schema to numpy data type

    Args:
        wrapped_type_strs (list[str]): the op schema type string

    Returns:
        list[np.dtype]: the converted numpy data type.
    """
    return [
        dtype
        for type_str in wrapped_type_strs
        if (dtype := try_convert_to_np_dtype(unwrap_type_str(type_str))) is not None
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


def get_full_operator_schemas() -> (
    list[tuple[str, list[tuple[int, list[tuple[str, defs.OpSchema, list[defs.OpSchema]]]]]]]
):
    """parse full operator schemas

    Returns:
        list[tuple[str, list[tuple[int, list[tuple[str, defs.OpSchema, list[defs.OpSchema]]]]]]]: nested structure containing all
            available op schemas
    """
    # domain -> support level -> name -> [schema]
    index: dict[str, dict[int, dict[str, list[defs.OpSchema]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for schema in defs.get_all_schemas_with_history():
        index[schema.domain][int(schema.support_level)][schema.name].append(schema)

    # Preprocess the Operator Schemas
    # [(domain, [(support_level, [(schema name, current schema, all versions schemas)])])]
    operator_schemas: list[tuple[str, list[tuple[int, list[tuple[str, defs.OpSchema, list[defs.OpSchema]]]]]]] = []
    existing_ops: set[str] = set()
    for domain, _supportmap in sorted(index.items()):
        if domain == defs.ONNX_ML_DOMAIN:
            continue

        processed_supportmap = []
        for _support, _namemap in sorted(_supportmap.items()):
            processed_namemap = []
            for n, unsorted_versions in sorted(_namemap.items()):
                versions = sorted(unsorted_versions, key=lambda s: s.since_version)
                schema = versions[-1]
                if schema.name in existing_ops:
                    continue
                existing_ops.add(schema.name)
                processed_namemap.append((n, schema, versions))
            processed_supportmap.append((_support, processed_namemap))
        operator_schemas.append((domain, processed_supportmap))
    return operator_schemas


def get_core_operator_schemas_defs() -> dict[str, defs.OpSchema]:
    """restructured operator schemas for only core operators

    Returns:
        dict[str, list[tuple[int, list[tuple[str, defs.OpSchema, list[defs.OpSchema]]]]]]: the dictionary with key-value pairs
            where each op name is a key in string whose value is the nest structure containing various properties
            of the ONNX op.
    """
    triples = dict(get_full_operator_schemas())[""][0][1]
    return {x[0]: x[1] for x in triples}


def get_core_operator_schemas() -> dict[str, OpSchema]:
    """restructured operator schemas for only core operators

    Returns:
        dict[str, list[tuple[int, list[tuple[str, defs.OpSchema, list[defs.OpSchema]]]]]]: the dictionary with key-value pairs
            where each op name is a key in string whose value is the nest structure containing various properties
            of the ONNX op.
    """
    triples = dict(get_full_operator_schemas())[""][0][1]
    return {x[0]: OpSchema.from_defs(x[1]) for x in triples}
