import json
import os
from abc import ABC, abstractmethod
from inspect import isclass
from typing import Any, ClassVar, TypeGuard

from packaging.version import Version
from typing_extensions import Self
from yacs.config import CfgNode

from ..owlite_core.logger import log


class Serializable(ABC):
    """Abstract protocol for version-aware serializable classes."""

    __available_since__: ClassVar[dict[str, Version] | None] = None

    @classmethod
    def truncate_version(cls, version: Version | None) -> Version | None:
        """Truncate the version from above by the current version if available.

        Args:
            version (Version | None): a target version or `None` to get the current version

        Returns:
            Version | None: the truncated version if the current version is available, `None` otherwise.
        """
        current_version: Version | None = getattr(cls, "__version__", None)
        if version is None:
            return current_version
        if current_version is not None and version > current_version:
            log.warning(
                f"The version {version} provided for {cls} is higher than current version {current_version}. "
                "Will use the current version instead"
            )
            return current_version
        return version

    @classmethod
    def check_version(cls, d: dict[str, Any]) -> None:
        """Check version compatibility.

        Args:
            d (dict[str, Any]): a dictionary containing values for initializing an instance of this class.

        Raises:
            TypeError: When class version doesn't match with data version
        """
        if isinstance((cls_version := getattr(cls, "__version__", None)), Version):
            d_version = Version(d.get("version", "1.0"))  # assume version 1.0 for data before format versioning

            if cls_version != d_version:
                raise TypeError(f"Version mismatch, cls({cls_version}) != data({d_version})")

    @classmethod
    def issubclass(cls, subclass: type[Any] | None) -> TypeGuard[Self]:
        """Check if the given class is a subclass of this class.

        Args:
            subclass (type[Any]): a class to check.

        Returns:
            TypeGuard[Self]: `True` if the class is a subclass of this class, `False` otherwise.
        """
        try:
            return isclass(subclass) and issubclass(subclass, cls)
        except TypeError:  # issubclass can raise type error when field.annotation is a type alias (e.g. list[str])
            return False

    @classmethod
    @abstractmethod
    def deserialize(cls, d: dict[str, Any] | str | Self) -> Self:
        """Load a serialized representation of an instance of this class.

        Args:
            d (dict[str, Any] | str | Self): a serialized representation of this class. Must be one of the followings:
                i) a python dictionary object representing an instance of this class; or
                ii) the string containing a such serialized representation; or
                iii) the path to a json or yaml file containing a such serialized representation; or
                iv) an instance of this class (in which case this function will just return the unmodified input.)

        Returns:
            Self: an instance of this class loaded from the input representation.
        """

    @classmethod
    def _deserialize(cls, d: dict[str, Any] | str | Self) -> dict[str, Any] | Self:
        if isinstance(d, cls):
            return d

        if isinstance(d, str):
            d = load_json_or_yaml(d)

        if not isinstance(d, dict):
            raise TypeError(f"{cls} cannot load invalid value {d} of type {type(d)}")

        cls.check_version(d)
        _ = d.pop("version", None)

        return d

    def serialize_as_yaml(self, version: Version | None = None) -> CfgNode:
        """Serialize this object as a `CfgNode`.

        You can use it for saving this object as a yaml file as follows:
        ```python
        with open("config.yaml", "w") as f:
            f.write(options.serialize_as_yaml().dump())
        ```.
        """
        return CfgNode(init_dict=self.serialize_as_json(version))

    @abstractmethod
    def serialize_as_json(self, version: Version | None = None) -> dict[str, Any]:
        """Serialize this object as a JSON-serializable dictionary.

        You can use it for saving this object as a json file as follows:
        ```python
        import json
        with open("config.json", "w") as f:
            json.dump(options.serialize_as_json(), f)
        ```.
        """


def load_json_or_yaml(path_or_string_literal: str) -> dict[str, Any]:
    """Load either json or CfgNode from the given string.

    Args:
        path_or_string_literal (str): a string object containing either
            * the path to a "*.json" or "*.yaml" file; or
            * the content of the file in string

    Returns:
        dict[str, Any]: the loaded object
    """
    try:
        if os.path.isfile(path_or_string_literal):
            with open(path_or_string_literal, encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(path_or_string_literal)
    except json.JSONDecodeError:
        if os.path.isfile(path_or_string_literal):
            with open(path_or_string_literal, encoding="utf-8") as f:
                data = CfgNode.load_cfg(f)
        else:
            data = CfgNode.load_cfg(path_or_string_literal)
        data = _convert_cfg_node_to_dict(data)

    if not isinstance(data, dict) or isinstance(data, CfgNode):
        raise TypeError(f"Expected dict, but {data} of type {type(data)} is loaded.")

    return data


def _convert_cfg_node_to_dict(node: CfgNode) -> dict[str, Any]:
    if not isinstance(node, CfgNode):
        return node
    return {k: _convert_cfg_node_to_dict(v) for k, v in node.items()}
