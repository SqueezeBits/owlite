from .op_schema import OpSchema, get_core_operator_schemas


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
        """The full schema object of the op

        Returns:
            OpSchema: the op schema
        """
        return ONNXOp.schemas[self.name]
