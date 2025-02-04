from enum import IntEnum


class PricePlan(IntEnum):
    """User's pricing plan."""

    UNKNOWN = 0
    FREE = 1
    CLOUD = 2
    ON_PREM = 3

    @property
    def paid(self) -> bool:
        """Whether the status indicates if the plan is paid."""
        return self > PricePlan.FREE
