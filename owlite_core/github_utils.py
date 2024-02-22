import requests

from .constants import OWLITE_GIT_REPO_URL


def get_latest_version_from_github() -> str:
    """Retrieves the latest release version of the package from GitHub.

    Args:
        repo_url(str): The url of the GitHub package.

    Returns:
        str: The latest release version if successful.

    Raises:
        requests.HTTPError: If the request to GitHub fails.
    """
    api_url = f"{OWLITE_GIT_REPO_URL}/releases/latest"
    response = requests.get(api_url, timeout=15)

    if not response.ok:
        response.raise_for_status()
    latest_release_version = response.url.split("/")[-1][1:]
    return latest_release_version
