import json
import os
from typing import Any

from packaging.version import Version
from yacs.config import CfgNode

from ..owlite_core.logger import log


def check_version(cls: Any, d: dict[str, Any]) -> None:
    """Check version compatibility.

    Args:
        cls (Any): `OptionsMixin` or `OptionsDict` class to load data to
        d (dict[str, Any]): the data to load

    Raises:
        TypeError: When class version doesn't match with data version
    """
    if isinstance((cls_version := getattr(cls, "__version__", None)), Version):
        d_version = Version(d.get("version", "1.0"))  # assume version 1.0 for data before format versioning

        if cls_version != d_version:
            if cls_version.major == d_version.major and cls_version.minor == d_version.minor:
                log.debug_warning(f"Micro version mismatch, cls({cls.__name__})({cls_version}) != data({d_version})")
            else:
                raise TypeError(f"Version mismatch, cls({cls.__name__})({cls_version}) != data({d_version})")


def load_json_or_yaml(path_or_string_literal: str) -> dict | CfgNode:
    """Load either json or CfgNode from the given string.

    Args:
        path_or_string_literal (str): a string object containing either
            * the path to a "*.json" or "*.yaml" file; or
            * the content of the file in string

    Returns:
        Union[dict, CfgNode]: the loaded object
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

    if not isinstance(data, dict | CfgNode):
        raise TypeError(f"Expected either dict or CfgNode, but {data} of type {type(data)} is loaded.")

    return data
