from typing import Any

from packaging.version import Version
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

from .options_dict import OptionsDict
from .serializable import Serializable


class OptionsMixin(
    BaseModel, Serializable, arbitrary_types_allowed=True, populate_by_name=True, validate_assignment=True
):
    """The Mixin-style base class for adding type-checking feature and custom value-checking feature."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        validate_assignment=True,
        validate_default=True,
        validate_return=True,
        extra="forbid",
    )

    @classmethod
    def deserialize(cls, d: dict | str | Self) -> Self:
        return cls.model_validate(cls._deserialize(d))

    def serialize_as_json(self, version: Version | None = None) -> dict[str, Any]:
        cls = type(self)
        version = cls.truncate_version(version)

        exclude: set[str] = set()
        if isinstance((available_since := cls.__available_since__), dict) and isinstance(version, Version):
            exclude = exclude.union(
                field
                for field in self.model_fields
                if isinstance((min_version := available_since.get(field)), Version) and version < min_version
            )
        d = self.model_dump(mode="json", exclude=exclude)
        if isinstance((cls_version := getattr(self, "__version__", None)), Version):
            d.update(version=str(version or cls_version))
        return d

    @model_validator(mode="before")
    @classmethod
    def preprocess_options_dict(cls, values: Any) -> dict[str, Any]:
        """Preprocess all fields of type `OptionsDict` before passing it to the Pydantic's type validator.

        Args:
            values (dict[str, Any]): a dictionary containing values for initializing an instance of this class.

        Returns:
            dict[str, Any]: the dictionary whose key corresponds to a field whose annotation is a subclass of
                `OptionsDict`.
        """
        for name, field in cls.model_fields.items():
            if not (isinstance(values, dict) and name in values and OptionsDict.issubclass(field.annotation)):
                continue
            values[name] = field.annotation.deserialize(values[name])
        return values
