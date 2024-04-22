from typing import Any, Generic, TypeVar, get_args

from packaging.version import Version
from pydantic import TypeAdapter
from typing_extensions import Self

from .serializable import Serializable

K = TypeVar("K", bound=str)
V = TypeVar("V")


class OptionsDict(dict[K, V], Generic[K, V], Serializable):
    """A simple extension of python `dict` to hold Options as values."""

    def __init__(self, d: dict[str, Any] | str | None = None):
        if self.key_type() is K or self.value_type() is V:  # type: ignore[misc]
            raise TypeError("You must specify key and value types of OptionsDict.")
        # Required for checking if the value type is valid
        super(dict, self).__init__()
        if d is None:
            return
        for k, v in type(self).deserialize(d).items():
            self[k] = v

    def update(self, d: dict[Any, Any]) -> None:  # type: ignore[override]
        for key, value in d.items():
            self[key] = value

    @classmethod
    def key_type(cls) -> type:
        """Get the key type of this dictionary."""
        # pylint: disable-next=no-member
        return get_args(cls.__orig_bases__[0])[0]  # type: ignore[attr-defined]

    @classmethod
    def value_type(cls) -> type:
        """Get the value type of this dictionary."""
        # pylint: disable-next=no-member
        return get_args(cls.__orig_bases__[0])[1]  # type: ignore[attr-defined]

    @classmethod
    def deserialize(cls, d: dict[str, Any] | str | Self) -> Self:
        d = cls._deserialize(d)

        options_dict = cls()
        for name, data in d.items():
            options_dict[name] = data  # type: ignore[index]

        return options_dict

    def serialize_as_json(self, version: Version | None = None) -> dict[str, Any]:
        ret = {
            name: (
                options.serialize_as_json(version)
                if isinstance(options, Serializable)
                else TypeAdapter(type(self).value_type()).dump_python(options)  # type: ignore[arg-type]
            )
            for name, options in self.items()
        }
        if isinstance((cls_version := getattr(self, "__version__", None)), Version):
            ret.update(version=str(self.truncate_version(version) or cls_version))
        return ret  # type: ignore[return-value]

    def __setitem__(self, key: K, value: V) -> None:
        key = TypeAdapter(type(self).key_type()).validate_python(key)
        value_type = type(self).value_type()
        if Serializable.issubclass(value_type):
            value = value_type.deserialize(value)  # type: ignore[assignment, arg-type]
        else:
            value = TypeAdapter(value_type).validate_python(value)
        return super().__setitem__(key, value)
