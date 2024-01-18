import json
import os
from typing import Union

from yacs.config import CfgNode


def load_json_or_yaml(path_or_string_literal: str) -> Union[dict, CfgNode]:
    """Loads either json or CfgNode from the given string.

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

    if not isinstance(data, (dict, CfgNode)):
        raise TypeError(f"Expected either dict or CfgNode, but {data} of type {type(data)} is loaded.")

    return data
