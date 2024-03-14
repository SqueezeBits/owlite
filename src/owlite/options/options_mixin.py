import inspect
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from typing import Any, Union, get_args, get_origin, get_type_hints

from typing_extensions import Self
from yacs.config import CfgNode

from ..owlite_core.logger import log
from .generic_type_checking import generic_isinstance, is_optional, unwrap_optional
from .load import check_version, load_json_or_yaml


class OptionsMixin:
    """The Mixin-style base class for adding type-checking feature and custom value-checking feature."""

    @classmethod
    def annotations(cls) -> dict[str, type]:
        """Finds the type inferred from the type hint of each member variable from a given class

        Args:
            cls (type): a class

        Returns:
            dict[str, type]: the type annotations
        """
        a = {}
        if cls.__base__ is not None and issubclass(cls.__base__, OptionsMixin):
            # pylint: disable-next=no-member
            a.update(cls.__base__.annotations())
        a.update(get_type_hints(cls))
        return a

    @classmethod
    def load(cls, d: Union[CfgNode, dict, str]) -> Self:
        """Loads the OptionsMixin subclass instance from the given data

        Args:
            d (Union[CfgNode, dict, str]): one of
                * a CfgNode or dict object; or
                * a string object containing the representation of such object; or
                * the path to a file containing such representation.

        Raises:
            KeyError: if a required key is not found

        Returns:
            Self: an OptionsMixin object
        """
        if isinstance(d, str):
            d = load_json_or_yaml(d)
        if not isinstance(d, dict):
            raise ValueError(f"{cls} cannot load invalid value {d}")

        check_version(cls, d)

        kwargs = {}
        if not is_dataclass(cls) or not issubclass(cls, OptionsMixin):
            log.error(f"A subclass of OptionsMixin must be decorated with dataclass, but {cls} is not")
            raise TypeError(f"{cls} must be decorated with dataclass")

        for field in fields(cls):
            required = field.default is MISSING and field.default_factory is MISSING
            default_value = None
            if field.default is not MISSING:
                default_value = field.default
            elif field.default_factory is not MISSING:
                default_value = field.default_factory()

            if required and field.name not in d:
                raise KeyError(f"Missing required key {field.name} in dictionary {d}")

            value = d.get(field.name, default_value)
            kwargs[field.name] = cls._deserialize(value, field.type)
        return cls(**kwargs)  # type: ignore[return-value]

    def __setattr__(self, name: str, value: Any) -> None:
        """Check the type of the new value.

        Args:
            name (str): the name of a property
            value (Any): the new value for the property

        Raises:
            KeyError: if `name` is not a pre-defined attribute.
            ValueError: if a method named `f"check_{name}"` is found,
                `self.check_{name}(value)` is evaluated, and if the
                result is False, raises ValueError with message including the
                method's doc string.
        """
        cls = self.__class__
        cls_name = cls.__name__
        annotations = cls.annotations()
        if name not in annotations:
            raise KeyError(f"No such property in {cls_name}: {name}")
        field_type = annotations[name]
        if field_type is not Any:
            value = self._deserialize(value, field_type)
        if not (field_type is Any or generic_isinstance(value, field_type)):
            raise ValueError(
                f"Expected a value of type {field_type}, "
                f"but received {value} of type {type(value)} for {name} in {cls_name}"
            )
        self._check(name, value)
        super().__setattr__(name, value)

    def _check(self, attr: str, new_value: Any) -> None:
        checker = getattr(self, f"check_{attr}", None)
        if checker and inspect.ismethod(checker) and not checker(new_value):
            msg = f"Invalid value {new_value} given to {attr} of {self.__class__.__name__}"
            doc_string = getattr(checker, "__doc__", None)
            if doc_string is not None:
                msg += f":\n{doc_string}"
            raise ValueError(msg)

    @property
    def config(self) -> CfgNode:
        """CfgNode representation for this object, which you can use it for writing it to a yaml file by
        ```python
        with open("config.yaml", "w") as f:
            f.write(options.config.dump())
        ```
        """
        return CfgNode(init_dict=self.json)

    @property
    def json(self) -> dict:
        """Builtin dictionary representation for this object, which you can use it for writing it to a json file
        with the `json.dump` or `json.dumps` function"""
        d = {}
        if version := getattr(self, "__version__", None):
            d["version"] = version
        for field in fields(type(self)):  # type: ignore[arg-type]
            field_value = getattr(self, field.name)
            d[field.name] = self._serialize(field_value)
        return d

    @classmethod
    def _deserialize(cls, x: object, t: type) -> object:
        type_error_message = (
            "A field type of OptionsMixin must"
            "\ni) be one of bool, int, float or str; or"
            "\nii) be a subclass of enum.Enum; or"
            "\niii) be a subclass of OptionsMixin; or"
            "\niv) be of the form Optional[T] where T satisfies one of i), ii) or iii)"
            "\nv) be of the form list[T] where T satisfies one of i), ii) or iii)"
            "\nvi) be of the form tuple[T, ...] where T satisfies one of i), ii) or iii)"
        )
        type_error_desc = f"Unsupported field type {t} found {cls}."
        if generic_isinstance(x, t):
            return x
        if is_optional(t):
            return None if x is None else cls._deserialize(x, unwrap_optional(t))
        args = get_args(t)
        origin = get_origin(t)
        if origin is list and isinstance(x, list):
            if len(args) != 1:
                log.error(type_error_message)
                raise TypeError(type_error_desc)
            element_type = args[0]
            return [cls._deserialize(item, element_type) for item in x]
        if origin is tuple and isinstance(x, (list, tuple)):
            if not (len(args) == 2 and args[1] is ...):
                log.error(type_error_message)
                raise TypeError(type_error_desc)
            element_type = args[0]
            return tuple(cls._deserialize(item, element_type) for item in x)
        if origin is not None:
            raise ValueError(f"Expected value of type {t}, but got {x}.")
        if issubclass(t, Enum):
            return t[x] if isinstance(x, str) else t(x)
        if issubclass(t, OptionsMixin) and isinstance(x, (dict, str)):
            return t.load(x)
        if t not in (int, float, str, bool):
            log.error(type_error_message)
            raise TypeError(type_error_desc)
        return t(x)  # type: ignore[call-arg]

    @classmethod
    def _serialize(cls, x: object) -> object:
        if isinstance(x, OptionsMixin):
            return x.json
        if isinstance(x, Enum):
            return x.name
        if isinstance(x, dict):
            return {key: cls._serialize(value) for key, value in x.items()}
        if isinstance(x, list):
            return [cls._serialize(value) for value in x]
        if isinstance(x, tuple):
            return tuple(cls._serialize(value) for value in x)
        return x
