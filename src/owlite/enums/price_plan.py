from enum import IntEnum


class PricePlan(IntEnum):
    """User's pricing plan."""

    FREE = 0
    LITE = 1
    BUSINESS = 2
    ON_PREM = 3

    @property
    def paid(self) -> bool:
        """Whether the status indicates if the plan is paid."""
        return self != PricePlan.FREE
