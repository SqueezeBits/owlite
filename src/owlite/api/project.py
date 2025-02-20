import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from requests.exceptions import HTTPError
from typing_extensions import Self

from ..core.api_base import MAIN_API_BASE
from ..core.cache.workspace import Workspace
from ..core.constants import OWLITE_FRONT_BASE_URL, OWLITE_HOME_PATH
from ..core.logger import log

if TYPE_CHECKING:
    from .baseline import Baseline


@dataclass
class Project:
    """The OwLite project."""

    workspace: Workspace
    id: str
    name: str
    baseline: Optional["Baseline"] = field(default=None)

    @property
    def url(self) -> str:
        """The URL to the project page."""
        return f"{OWLITE_FRONT_BASE_URL}/project/detail?workspace_id={self.workspace.id}&project_id={self.id}"

    @property
    def home(self) -> str:
        """The directory path for writing outputs produced by this project."""
        return str(OWLITE_HOME_PATH / self.name)

    @classmethod
    def create(cls, workspace: Workspace, name: str, description: str | None = None) -> Self:
        """Create a new project.

        Args:
            workspace (Workspace): The workspace to create the project in
            name (str): The name for the project to be created
            description (str | None, optional): Optional description for the project. Defaults to None.

        Raises:
            RuntimeError: When the project is not created for an unexpected reason.

        Returns:
            Project: The newly created project.
        """
        if description is None:
            description = ""
        resp = MAIN_API_BASE.post(
            "/projects",
            json={
                "workspace_id": workspace.id,
                "project_name": name,
                "description": description,
            },
        )

        if not (isinstance(resp, dict) and resp["name"] == name):
            raise RuntimeError(f"Failed to create project '{name}'")

        project = cls(workspace, resp["id"], resp["name"])
        log.info(f"Created a new {project}")  # UX
        return project

    @classmethod
    def load_or_create(cls, workspace: Workspace, name: str, description: str | None = None) -> Self:
        """Load the existing project named `name` if found, creates a new one otherwise.

        Args:
            workspace (Workspace): The workspace to load the project in
            name (str): The name of the project to be loaded or created
            description (str | None, optional): Optional description that will be used only when a new project is
                created. Defaults to None.

        Raises:
            e (HTTPError): When an unexpected HTTP status code is returned.

        Returns:
            Project: the loaded or created project
        """
        try:
            return cls.create(workspace, name, description)
        except HTTPError as e:
            if e.response is not None and e.response.status_code == 409:  # the project already exists
                data = json.loads(e.response.content)
                assert (
                    data["detail"]["existing_project_name"] == name
                ), f"Project name mismatch: {data['detail']['existing_project_name']} != {name}"
                project = cls(workspace, data["detail"]["existing_project_id"], name)
                log.info(f"Loaded the existing {project}")  # UX
                return project

            if e.response is not None and e.response.status_code == 403:
                log.error(
                    "You can create up to 2 Projects in a single Free Plan Workspace. "
                    "In this execution, OwLite functions will not be executed. "
                    "Please delete an existing Project or register it in an existing one."
                )  # UX
            raise e

    def __str__(self) -> str:
        return f"project '{self.name}'"
