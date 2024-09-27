"""OwLite Base URL Management Module.

This module handles the caching of base URLs used in OwLite APIs.
"""

from ...core.logger import log
from ..device_settings import OWLITE_DEVICE_SETTINGS
from ..settings import OWLITE_SETTINGS
from .device import disconnect_device

URL_NAME_LIST = ["FRONT", "MAIN", "DOVE", "NEST"]


def save_base_url(name: str, url: str) -> None:
    """Save the base URL for an API in the cache.

    Args:
        name (str): A name of a URL.
        url (str): URL to save.

    Raises:
        ValueError: If the API name is invalid.
    """
    if name not in URL_NAME_LIST:
        log.error(f"Invalid API base name: '{name}'. Valid API base names are {URL_NAME_LIST}")  # UX
        raise ValueError(f"Invalid value given to url name: {name}")
    base_urls = OWLITE_SETTINGS.base_url
    base_urls.set(name, url)
    OWLITE_SETTINGS.base_url = base_urls

    if (
        name == "NEST"
        and OWLITE_DEVICE_SETTINGS.connected
        and OWLITE_DEVICE_SETTINGS.connected.manager.name == name
        and OWLITE_DEVICE_SETTINGS.connected.manager.url != url
    ):
        disconnect_device()

    log.info(f"The {name} API base is set to {url}")  # UX


def print_base_urls() -> None:
    """Print base url in cache."""
    base_urls = OWLITE_SETTINGS.base_url
    url_list = "\n".join([f"{name} : {getattr(base_urls, name)}" for name in URL_NAME_LIST])
    log.info(f"Base urls list\n{url_list}")  # UX


def delete_base_url(name: str) -> None:
    """Delete url in cache.

    Args:
        name (str): base url's name
    """
    if name not in URL_NAME_LIST:
        log.error(f"Invalid API base name: '{name}'. Valid API base names are {URL_NAME_LIST}")  # UX
        raise ValueError(f"Invalid value given to url name: {name}")

    base_urls = OWLITE_SETTINGS.base_url
    base_urls.set(name)
    OWLITE_SETTINGS.base_url = base_urls
    log.info(f"Deleted the {name} API base")  # UX

    if name == "NEST" and OWLITE_DEVICE_SETTINGS.connected and OWLITE_DEVICE_SETTINGS.connected.manager.name == name:
        disconnect_device()
