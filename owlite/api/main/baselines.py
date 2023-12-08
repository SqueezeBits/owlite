import requests

from ..api_base import MAIN_API_BASE


def create_baseline(project_id: str, baseline_name: str) -> str:
    """Creates a baseline experiment with given baseline name at project with given project id.

    Args:
        project_id (str): The id of a project.
        baseline_name (str): The name of a baseline to create.

    Returns:
        str: The name of created baseline.

    Raises:
        HTTPError: When request was not successful.
    """
    payload = {
        "project_id": project_id,
        "baseline_name": baseline_name,
    }

    resp = MAIN_API_BASE.post("/projects/baselines", json=payload)
    assert isinstance(resp, dict)

    return resp["baseline_name"]


def check_baseline_existence(project_id: str, baseline_name: str) -> bool:
    """Checks if baseline with given name exists at project with given project id.

    Args:
        project_id (str): The id of a project.
        baseline_name (str): The name to check.

    Returns:
        bool: True if baseline exists in given project, False otherwise.

    Raises:
        HTTPError: When request was not successful.
    """
    payload = {
        "project_id": project_id,
        "baseline_name": baseline_name,
    }

    try:
        resp = MAIN_API_BASE.post("/projects/baselines/check", json=payload)
        assert isinstance(resp, bool)

        return resp

    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return False

        raise e
