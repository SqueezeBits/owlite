"""OwLite Base URL Management Module

This module handles the caching of base URLs used in OwLite APIs."""

from ..logger import log
from ..owlite_settings import OWLITE_SETTINGS

URL_NAME_LIST = ["FRONT", "MAIN", "DOVE"]


def save_base_url(name: str, url: str) -> None:
    """Saves url in cache.

    Args:
        name (str): A name of a URL.
        url (str): URL to save.

    Raises:
        HTTPError: When login request was not successful.
    """
    if name not in URL_NAME_LIST:
        log.error(f"Use name in the list: {URL_NAME_LIST}")
        raise ValueError(f"Invalid value given to url name: {name}")

    OWLITE_SETTINGS.base_url = {"name": name, "url": url}

    log.info(f"{name} API Base URL set to {url}")


def print_base_urls() -> None:
    """Prints base url in cache."""
    base_url_dict = OWLITE_SETTINGS.base_url
    url_list = "\n".join([f"{name} : {url}" for name, url in base_url_dict.items()])
    log.info(f"Base urls list\n{url_list}")


def delete_base_url(name: str) -> None:
    """Deletes url in cache.

    Args:
        name (str): base url's name
    """
    if name not in URL_NAME_LIST:
        log.error(f"Use name in the list: {URL_NAME_LIST}")
        raise ValueError(f"Invalid value given to url name: {name}")

    OWLITE_SETTINGS.base_url = {"name": name}
    log.info(f"Deleted the url '{name}'")
