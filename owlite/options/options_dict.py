"""Options required for configuring torch.fx.GraphModule"""
from dataclasses import fields, is_dataclass
from types import NoneType
from typing import Any, Union, get_args, get_origin

from yacs.config import CfgNode

from ..logger import log
from .generic_type_checking import generic_isinstance
from .load import load_json_or_yaml
from .options_mixin import OptionsMixin


class OptionsDict(dict, OptionsMixin):
    """A simple extension of python `dict` to hold Options as values"""

    ValueType: type

    def __init__(self, d: Union[CfgNode, dict, str, NoneType] = None):
        # Required for checking if ValueType is valid
        _ = type(self).value_types()
        super(dict, self).__init__()
        if d is None:
            return
        for k, v in type(self).load(d).items():
            self[k] = v

    def update(self, d: dict):
        for key, value in d.items():
            self[key] = value

    @classmethod
    def value_types(cls) -> tuple[type[OptionsMixin]]:
        """Allowed value types of this class in tuple"""
        if hasattr(cls, "_value_types"):
            # prevent duplicate type-checking
            return cls._value_types
        if not hasattr(cls, "ValueType"):
            log.error(
                "A subclass of OptionsDict requires a static type (or type union) `ValueType` "
                "indicating the possible value types of the subclass"
            )
            raise AttributeError(f"ValueType for {cls} is not defined")

        origin = get_origin(cls.ValueType)
        full_type_error_message = (
            f"The type (union) ValueType of {cls} must be one of the followings:"
            "\ni) a subclass of OptionsMixin decorated with dataclass; or"
            "\nii) the list, tuple, Optional or Union of type(s) satisfying i); or"
            "\niii) a subclass of OptionsDict,\n"
            f"but {cls.__name__}.ValueType={cls.ValueType} is given."
        )
        type_error_message = f"Invalid ValueType {cls.ValueType} defined for {cls}"
        if origin in (Union, list, tuple):
            args = get_args(cls.ValueType)
            if not all((issubclass(c, OptionsMixin) and is_dataclass(c)) or c is NoneType for c in args):
                log.error(full_type_error_message)
                raise TypeError(type_error_message)
        elif origin is None:
            if not (
                (issubclass(cls.ValueType, OptionsMixin) and is_dataclass(cls.ValueType))
                or issubclass(cls.ValueType, OptionsDict)
            ):
                log.error(full_type_error_message)
                raise TypeError(type_error_message)
            args = (cls.ValueType,)
        else:
            raise TypeError(f"The type hint origin {origin} is not supported - {cls}.ValueType = {cls.ValueType}")
        cls._value_types = args
        return args

    @classmethod
    def load(cls, d: Union[dict, list, str, tuple, NoneType]) -> Any:
        options_dict = cls()
        value_types = cls.value_types()
        origin = get_origin(cls.ValueType)

        def load(name: str, data):
            if generic_isinstance(data, cls.ValueType):
                options_dict[name] = data
                return
            if origin in (Union, None):
                if data is None and NoneType in value_types:
                    options_dict[name] = None
                    return
                if not isinstance(data, dict):
                    raise TypeError(f"Expected dict but got {data}")
                subnode_key_set = set(data.keys())
                for option_type in value_types:
                    if issubclass(option_type, OptionsDict):
                        options_dict[name] = option_type.load(data)
                        break
                    if subnode_key_set == {field.name for field in fields(option_type)}:
                        options_dict[name] = option_type.load(data)
                        break
                else:
                    raise ValueError(
                        f"Failed to parse config for node {name}: "
                        f"no matching options class for {data}. ({cls.__name__}.ValueType = {cls.ValueType})"
                    )
                return
            if origin in (list, tuple) and isinstance(data, (tuple, list)):
                if len(value_types) != 1:
                    raise TypeError(
                        "When ValueType of a subclass of OptionsDict is either list or tuple, "
                        f"its element type must be specified, but {cls.__name__}.ValueType = {cls.ValueType}"
                    )
                option_type = value_types[0]
                options_dict[name] = origin(option_type.load(item) for item in data)
                return
            raise ValueError(f"{cls} cannot load the invalid value {data} at key={name}")

        if isinstance(d, str):
            d = load_json_or_yaml(d)

        for name, data in d.items():
            load(name, data)

        return options_dict

    @property
    def config(self) -> CfgNode:
        """CfgNode representation for this object, which you can use it for writing it to a yaml file by
        ```python
        with open("config.yaml", "w") as f:
            f.write(options.config.dump())
        ```
        """
        return CfgNode(self.json)

    @property
    def json(self) -> dict:
        """Builtin dictionary representation for this object, which you can use it for writing it to a json file
        with the `json.dump` or `json.dumps` function"""
        origin = get_origin(type(self).ValueType)
        if origin in (tuple, list):
            return {name: origin(item.json for item in options) for name, options in self.items()}
        return {
            name: (options.json if isinstance(options, OptionsMixin) else options) for name, options in self.items()
        }

    def __setitem__(self, key: Any, value: Any) -> None:
        if not generic_isinstance(value, type(self).ValueType):
            raise ValueError(
                f"{type(self)} expected value of type {type(self).ValueType}, "
                f"but {value} of type {type(value)} is given"
            )
        return super().__setitem__(key, value)
