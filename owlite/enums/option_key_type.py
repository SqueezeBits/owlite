from enum import Enum


class ModuleInsertionPoint(Enum):
    """The positions where a call-module node can be inserted"""

    INPUT_NODES = 0
    ARGS = 1
    KWARGS = 2
