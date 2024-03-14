from dataclasses import fields, is_dataclass
from types import NoneType
from typing import Any, Generic, Optional, TypeVar, Union, get_args, get_origin

from yacs.config import CfgNode

from ..owlite_core.logger import log
from .generic_type_checking import generic_isinstance
from .load import check_version, load_json_or_yaml
from .options_mixin import OptionsMixin

K = TypeVar("K")
V = TypeVar("V")


class OptionsDict(dict[K, V], Generic[K, V], OptionsMixin):
    """A simple extension of python `dict` to hold Options as values"""

    def __init__(self, d: Optional[Union[dict, str]] = None):
        # Required for checking if the value type is valid
        _ = type(self).value_type_origins()
        super(dict, self).__init__()
        if d is None:
            return
        for k, v in type(self).load(d).items():
            self[k] = v

    def update(self, d: dict[Any, Any]) -> None:  # type: ignore[override]
        for key, value in d.items():
            self[key] = value

    @classmethod
    def key_type(cls) -> type:
        """The key type of this dictionary"""
        # pylint: disable-next=no-member
        return get_args(cls.__orig_bases__[0])[0]  # type: ignore[attr-defined]

    @classmethod
    def value_type(cls) -> type:
        """The value type of this dictionary"""
        # pylint: disable-next=no-member
        return get_args(cls.__orig_bases__[0])[1]  # type: ignore[attr-defined]

    @classmethod
    def value_type_origins(cls) -> tuple[type[OptionsMixin], ...]:
        """Allowed value types of this class in tuple"""
        if hasattr(cls, "_value_type_origins"):
            # prevent duplicate type-checking
            return cls._value_type_origins

        key_type = cls.key_type()
        value_type = cls.value_type()

        if key_type is K or value_type is V:  # type: ignore[misc]
            raise TypeError("OptionsDict is used without proper key and value type specialization.")

        args: tuple[Any, ...] = get_args(value_type)
        origin = get_origin(value_type)
        full_type_error_message = (
            f"The value type of {cls} must be one of the followings:"
            "\ni) a builtin type (bool, float, int and str); or"
            "\nii) a subclass of OptionsMixin decorated with dataclass; or"
            "\niii) the list, variadic tuple, Optional or Union of type(s) satisfying i) or ii); or"
            "\niv) a subclass of OptionsDict,\n"
            f"but {cls.__name__} got value type {value_type}."
        )
        type_error_message = f"Invalid value type {value_type} defined for {cls}"

        def is_valid_element_type(c: type, include_nonetype: bool = True) -> bool:
            if include_nonetype and c is NoneType:
                return True
            return (issubclass(c, OptionsMixin) and is_dataclass(c)) or c in (bool, float, int, str)

        if origin is None:
            if not (is_valid_element_type(value_type, include_nonetype=False) or issubclass(value_type, OptionsDict)):
                log.error(full_type_error_message)
                raise TypeError(type_error_message)
            args = (value_type,)
        elif not (
            (origin is Union and all(is_valid_element_type(c) for c in args))
            or (origin is list and len(args) == 1 and is_valid_element_type(args[0]))
            or (origin is tuple and len(args) == 2 and is_valid_element_type(args[0]) and args[1] is ...)
        ):
            if origin is list:
                raise TypeError(
                    f"Missing element type in the list type hint - {cls.__name__} got value type {value_type}"
                )
            if origin is tuple:
                raise TypeError(
                    "Only variadic tuple (e.g. tuple[T, ...]) is supported for type hint, "
                    f"but {cls.__name__} got value type {value_type}"
                )
            if origin is Union:
                raise TypeError(
                    "Only Optional or Union of subclasses of OptionsMixin wrapped with dataclass "
                    "or builtin types (bool, float, int, str) is supported, "
                    f"but {cls.__name__} got value type {value_type}"
                )
            raise TypeError(
                f"The type hint origin {origin} is not supported - {cls.__name__} got value type {value_type}"
            )
        cls._value_type_origins = args  # type: ignore[attr-defined]
        return args

    @classmethod
    def load(cls, d: Union[dict, str]) -> Any:
        options_dict = cls()
        value_type_origins = cls.value_type_origins()
        value_type = cls.value_type()
        origin = get_origin(value_type)

        def load(name: K, data: Any) -> None:
            if generic_isinstance(data, value_type):
                options_dict[name] = data
                return
            if origin in (Union, None):
                if data is None and NoneType in value_type_origins:  # type: ignore[comparison-overlap]
                    options_dict[name] = None  # type: ignore[assignment]
                    return
                if not isinstance(data, dict):
                    raise TypeError(f"Expected dict but got {data}")
                subnode_key_set = set(data.keys())
                for option_type in value_type_origins:
                    if issubclass(option_type, OptionsDict):
                        options_dict[name] = option_type.load(data)
                        break
                    if subnode_key_set == {field.name for field in fields(option_type)}:  # type: ignore[arg-type]
                        options_dict[name] = option_type.load(data)  # type: ignore[assignment]
                        break
                else:
                    raise ValueError(
                        f"Failed to parse config for node {name}: "
                        f"no matching options class for {data}. ({cls.__name__} got value type {value_type})"
                    )
                return
            if origin in (list, tuple) and isinstance(data, (tuple, list)):
                if len(value_type_origins) != (1 if origin is list else 2):
                    raise TypeError(
                        "When the value type of a subclass of OptionsDict is either list or a variadic tuple, "
                        f"its element type must be specified, but {cls.__name__} got value type {value_type}"
                    )
                option_type = value_type_origins[0]
                options_dict[name] = origin(option_type.load(item) for item in data)
                return
            raise ValueError(f"{cls} cannot load the invalid value {data} at key={name}")

        if isinstance(d, str):
            d = load_json_or_yaml(d)

        check_version(cls, d)

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
        origin = get_origin(type(self).value_type())

        ret = {
            name: origin(item.json for item in options)  # type: ignore[attr-defined]
            if origin in (tuple, list)
            else (options.json if isinstance(options, OptionsMixin) else options)
            for name, options in self.items()
        }
        if version := getattr(self, "__version__", None):
            ret.update(version=version)

        return ret

    def __setitem__(self, key: K, value: V) -> None:
        if not generic_isinstance(key, type(self).key_type()):
            raise ValueError(
                f"{type(self)} expected a key of type {type(self).key_type()}, but {key} of type {type(key)} is given"
            )
        if not generic_isinstance(value, type(self).value_type()):
            raise ValueError(
                f"{type(self)} expected a value of type {type(self).value_type()}, "
                f"but {value} of type {type(value)} is given"
            )
        return super().__setitem__(key, value)
