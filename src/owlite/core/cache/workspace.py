import requests
from pydantic import BaseModel
from typing_extensions import Self

from ...enums import PricePlan


class Workspace(BaseModel):
    """Represents the workspace.

    Attributes:
        id (str): The ID of the workspace.
        name (str): The name of the workspace.
        plan (PricePlan): The price plan of the workspace.
    """

    id: str
    name: str
    plan: PricePlan

    @classmethod
    def load(cls, workspace_id: str) -> Self:
        """Load the workspace with the given id.

        Args:
            workspace_id (str): The id of the workspace to load.

        Returns:
            Workspace: The loaded workspace.
        """
        from ..api_base import MAIN_API_BASE  # pylint: disable=import-outside-toplevel

        try:
            resp = MAIN_API_BASE.get(f"/workspaces/{workspace_id}")
        except requests.exceptions.HTTPError as e:
            raise e

        assert isinstance(resp, dict)

        return cls.model_validate(resp)
