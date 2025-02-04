import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from requests.exceptions import HTTPError
from typing_extensions import Self

from ..core.api_base import MAIN_API_BASE
from ..core.constants import OWLITE_FRONT_BASE_URL, OWLITE_HOME_PATH
from ..core.logger import log

if TYPE_CHECKING:
    from .baseline import Baseline


@dataclass
class Project:
    """The OwLite project."""

    id: str
    name: str
    baseline: Optional["Baseline"] = field(default=None)

    @property
    def url(self) -> str:
        """The URL to the project page."""
        return f"{OWLITE_FRONT_BASE_URL}/project/detail/{self.id}"

    @property
    def home(self) -> str:
        """The directory path for writing outputs produced by this project."""
        return str(OWLITE_HOME_PATH / self.name)

    @classmethod
    def create(cls, name: str, description: str | None = None) -> Self:
        """Create a new project.

        Args:
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
                "project_name": name,
                "description": description,
            },
        )

        if not (isinstance(resp, dict) and resp["name"] == name):
            raise RuntimeError(f"Failed to create project '{name}'")

        project = cls(resp["id"], resp["name"])
        log.info(f"Created a new {project}")  # UX
        return project

    @classmethod
    def load_or_create(cls, name: str, description: str | None = None) -> Self:
        """Load the existing project named `name` if found, creates a new one otherwise.

        Args:
            name (str): The name of the project to be loaded or created
            description (str | None, optional): Optional description that will be used only when a new project is
                created. Defaults to None.

        Raises:
            e (HTTPError): When an unexpected HTTP status code is returned.

        Returns:
            Project: the loaded or created project
        """
        try:
            return cls.create(name, description)
        except HTTPError as e:
            if e.response is not None and e.response.status_code == 409:  # the project already exists
                data = json.loads(e.response.content)
                project = cls(data["detail"], name)
                log.info(f"Loaded the existing {project}")  # UX
                return project

            if e.response is not None and e.response.status_code == 403:
                log.error(
                    "Free Plan users can create up to two Projects. In this execution, OwLite functions "
                    "will not be executed. Please delete an existing Project or register it in an existing one."
                )  # UX
            raise e

    def __str__(self) -> str:
        return f"project '{self.name}'"
