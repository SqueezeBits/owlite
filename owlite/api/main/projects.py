"""API wrapper module for projects"""

import json

from requests.exceptions import HTTPError

from ...logger import log
from ..api_base import MAIN_API_BASE


def create_or_load_project(project_name: str, description: str = "") -> str:
    """Creates a project with given name and description and return the id of created project, if
    a project with given name already exists and accessible by current user, return the id of
    existing project.

    Args:
        project_name (str): The name of a project.
        description (str): The description of a project. Defaults to "".

    Returns:
        str: The id of a created project.

    Raises:
        HTTPError: When request was not successful.
    """
    log.debug(f"Attempt creating project with name {project_name}.")

    payload = {
        "project_name": project_name,
        "description": description,
    }

    try:
        resp = MAIN_API_BASE.post("/projects", json=payload)

        assert isinstance(resp, dict) and resp["name"] == project_name

        log.info(f"Created new project '{project_name}'")
        return resp["id"]

    except HTTPError as err:
        if err.response is not None and err.response.status_code == 409:
            # project with given name already was created by user before

            data = json.loads(err.response.content)
            project_id = data["detail"]

            log.debug(f"Conflict detected, project with name {project_name} already exists, loading existing project.")
            log.info(f"Loaded existing project '{project_name}'")
            return project_id

        raise err
