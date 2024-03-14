from enum import Enum


class ParamStatus(Enum):
    """The three possible statuses of an input parameter of `forward` method after tracing"""

    ALIVE = 0
    KEPT = 1
    PURGED = 2
