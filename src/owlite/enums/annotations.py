from collections.abc import Callable
from enum import Enum, IntEnum
from typing import TypeVar

# pylint: disable-next=invalid-name
IntEnumType = TypeVar("IntEnumType", bound=IntEnum)


def get_before_validator(int_enum_type: type[IntEnumType]) -> Callable[[int | str | IntEnumType], IntEnumType]:
    """Get a before validator for the given `IntEnum` subclass that converts integer or string value to it.

    Args:
        int_enum_type (type[IntEnumType]): a subclass of `IntEnum`.

    Returns:
        Callable[[int | str | IntEnumType], IntEnumType]: the function that converts either an integer representing an
            enum value or a string representing the name of a enum category
    """

    def preprocess(value: int | str | IntEnumType) -> IntEnumType:
        if isinstance(value, int_enum_type):
            return value
        if isinstance(value, str):
            return int_enum_type[value]
        return int_enum_type(value)

    return preprocess


def serialize_as_name(enum: Enum) -> str:
    """Return the name of the given `Enum` object."""
    return enum.name
