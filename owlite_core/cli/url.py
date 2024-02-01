"""OwLite Base URL Management Module

This module handles the caching of base URLs used in OwLite APIs."""

from owlite_core.logger import log

from ..owlite_settings import OWLITE_SETTINGS

URL_NAME_LIST = ["FRONT", "MAIN", "DOVE", "NEST"]


def save_base_url(name: str, url: str) -> None:
    """Saves url in cache.

    Args:
        name (str): A name of a URL.
        url (str): URL to save.

    Raises:
        HTTPError: When login request was not successful.
    """
    if name not in URL_NAME_LIST:
        log.error(f"Invalid API base name: '{name}'. Valid API base names are {URL_NAME_LIST}")
        raise ValueError(f"Invalid value given to url name: {name}")
    base_urls = OWLITE_SETTINGS.base_url
    base_urls.set(name, url)
    OWLITE_SETTINGS.base_url = base_urls

    log.info(f"The {name} API base is set to {url}")


def print_base_urls() -> None:
    """Prints base url in cache."""
    base_urls = OWLITE_SETTINGS.base_url
    url_list = "\n".join([f"{name} : {getattr(base_urls, name)}" for name in URL_NAME_LIST])
    log.info(f"Base urls list\n{url_list}")


def delete_base_url(name: str) -> None:
    """Deletes url in cache.

    Args:
        name (str): base url's name
    """
    if name not in URL_NAME_LIST:
        log.error(f"Invalid API base name: '{name}'. Valid API base names are {URL_NAME_LIST}")
        raise ValueError(f"Invalid value given to url name: {name}")

    base_urls = OWLITE_SETTINGS.base_url
    base_urls.set(name)
    OWLITE_SETTINGS.base_url = base_urls
    log.info(f"Deleted the {name} API base")
