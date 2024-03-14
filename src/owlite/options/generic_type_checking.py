from types import NoneType
from typing import Any, Union, get_args, get_origin


def generic_isinstance(obj: Any, type_hint: Union[type, tuple[type]]) -> bool:
    """An extension for the builtin function `isinstance` for type hint checking."""
    if isinstance(type_hint, tuple):
        return any(generic_isinstance(obj, t) for t in type_hint)

    origin_type = getattr(type_hint, "__origin__", None)
    if origin_type is None:
        return isinstance(obj, type_hint)
    value_types = get_args(type_hint)
    if origin_type is dict:
        value_type = value_types[0]
        return isinstance(obj, origin_type) and all(generic_isinstance(x, value_type) for x in obj.values())
    if origin_type in (tuple, list):
        value_type = value_types[0]
        return isinstance(obj, origin_type) and all(generic_isinstance(x, value_type) for x in obj)
    if origin_type is Union:
        return generic_isinstance(obj, value_types)
    raise NotImplementedError(f"generic_isinstance for {type_hint} is not implemented.")


def generic_issubclass(type_hint: type, superclass: Union[type, tuple[type]]) -> bool:
    """An extension for the builtin function `issubclass` for type hint checking."""
    if isinstance(superclass, tuple):
        return any(generic_issubclass(type_hint, s) for s in superclass)

    origin_type = getattr(type_hint, "__origin__", None)
    if origin_type is None:
        return issubclass(type_hint, superclass)
    if origin_type in (dict, tuple, list):
        return issubclass(type_hint, superclass)
    if origin_type is Union:
        field_type_args = get_args(type_hint)
        return any(generic_issubclass(x, superclass) for x in field_type_args)
    raise NotImplementedError(f"generic_issubclass for {type_hint} is not implemented.")


def is_optional(type_hint: type) -> bool:
    """Checks if the type hint is wrapped with Optional"""
    if get_origin(type_hint) is not Union:
        return False
    args = get_args(type_hint)
    return len(args) == 2 and NoneType in args


def unwrap_optional(type_hint: type) -> type:
    """Unwraps the Optional from the type hint if it is wrapped"""
    if not is_optional(type_hint):
        return type_hint
    return [arg for arg in get_args(type_hint) if arg is not NoneType][0]
