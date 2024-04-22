from enum import IntEnum


class ForwardParamStatus(IntEnum):
    """The three possible statuses of an input parameter of `forward` method after tracing."""

    ALIVE = 0
    KEPT = 1
    PURGED = 2
