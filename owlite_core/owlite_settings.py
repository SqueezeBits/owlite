import json
from pathlib import Path
from typing import Optional, cast

from . import owlite_cache_dir, read_text
from .constants import (
    OWLITE_DEFAULT_DEVICE_MANAGER,
    OWLITE_DOVE_API_BASE_URL,
    OWLITE_FRONT_BASE_URL,
    OWLITE_MAIN_API_BASE_URL,
)


class OwLiteSettings:
    """Handles OwLite settings including token management.

    OwLiteSettings manages tokens and URLs within the OwLite system.
    It provides methods to retrieve and store tokens for authentication.

    Attributes:
        path_tokens (Path): Path to store token information.
        path_url (Path): Path to store URL information.
    """

    def __init__(self) -> None:
        """Initialize OwLite settings.

        Initialize paths for OwLite cache directory to store tokens and URLs.
        """
        Path(owlite_cache_dir).mkdir(parents=True, exist_ok=True)

        self.path_tokens = Path(owlite_cache_dir) / "tokens"
        self.path_devices = Path(owlite_cache_dir) / "devices"
        self.path_connected = Path(owlite_cache_dir) / "connected"
        self.path_url = Path(owlite_cache_dir) / "urls"

    @property
    def tokens(self) -> Optional[dict[str, str]]:
        """Retrieves a token or None if it doesn't exist.

        Returns:
            Optional[dict[str,str]]: dict of access token and refresh token, None if the tokens don't exist.
        """
        read_tokens = read_text(self.path_tokens)
        if not read_tokens:
            return None
        return cast(dict[str, str], json.loads(read_tokens))

    @tokens.setter
    def tokens(self, new_tokens: Optional[dict[str, str]]) -> None:
        """Set new tokens or remove existing tokens.

        Args:
            new_tokens (Optional[dict[str, str]]): Dictionary containing access token and refresh token.
                If None, existing tokens will be removed.
        """
        if new_tokens:
            self.path_tokens.write_text(
                json.dumps(
                    {
                        "access_token": new_tokens["access_token"],
                        "refresh_token": new_tokens["refresh_token"],
                    }
                ),
                encoding="utf-8",
            )
        else:
            self.path_tokens.unlink(missing_ok=True)

    @property
    def managers(self) -> dict[str, str]:
        """Retrieves the device manager dictionary or None if it doesn't exist.

        Returns:
            dict[str, str]: Device manager dictionary,
        """
        cached_managers = read_text(self.path_devices)
        default_manager = {"DEFAULT": OWLITE_DEFAULT_DEVICE_MANAGER}
        if cached_managers:
            return dict(default_manager, **cast(dict[str, str], json.loads(cached_managers)))
        return default_manager

    @managers.setter
    def managers(self, manager: dict[str, str]) -> None:
        """Saves a device manager or removes a device manager

        Args:
            name (str): The name of the device manager.
            url (str): The url of the device manager.
        """
        if "url" in manager:
            manager_dict = self.managers
            manager_dict[manager["name"]] = manager["url"]
            self.path_devices.write_text(json.dumps(manager_dict), encoding="utf-8")
        else:
            manager_dict = self.managers
            manager_dict.pop(manager["name"], None)
            self.path_devices.write_text(json.dumps(manager_dict), encoding="utf-8")

    @property
    def connected_device(self) -> Optional[dict[str, str]]:
        """Retrieves the connected device.

        Returns:
            dict[str, str], optional: A dict containing the device manager's name, url and selected device,
            or None if no device is selected.
        """
        connected_device = read_text(self.path_connected)
        if connected_device:
            return cast(dict[str, str], json.loads(connected_device))
        return None

    @connected_device.setter
    def connected_device(self, device: Optional[dict[str, str]] = None) -> None:
        """Connects to the device manager and selects device or deletes a device setting from storage.

        Does not fail if the device does not exist.

        Args:
            device (dict[str, str]): The name, url, device of the device manager to connect.
        """
        if device:
            self.path_connected.write_text(json.dumps(device), encoding="utf-8")
        else:
            self.path_connected.unlink(missing_ok=True)

    @property
    def base_url(self) -> dict[str, str]:
        """Retrieves base URLs.

        Returns the base URLs including FRONT, MAIN, and DOVE if available.
        If no custom URLs are set, default OwLite base URLs are returned.

        Returns:
            dict[str, str]: Dictionary containing base URLs.
        """
        default_urls = {
            "FRONT": OWLITE_FRONT_BASE_URL,
            "MAIN": OWLITE_MAIN_API_BASE_URL,
            "DOVE": OWLITE_DOVE_API_BASE_URL,
        }
        base_urls = read_text(self.path_url)
        if not base_urls:
            return default_urls
        url_dict = json.loads(base_urls)
        return dict(default_urls, **url_dict)

    @base_url.setter
    def base_url(self, base_url: dict[str, str]) -> None:
        """Set or remove custom base URLs.

        Args:
            base_url (dict[str, str]):
                Dictionary containing the 'name' and 'url' keys to set or remove custom base URLs.

        Raises:
            ValueError: If the provided 'base_url' dictionary is invalid or incomplete.
        """

        if "url" in base_url:
            url_dict = self.base_url
            url_dict[base_url["name"]] = base_url["url"]
            self.path_url.write_text(json.dumps(url_dict))
        else:
            url_dict = self.base_url
            if not url_dict:
                return
            url_dict.pop(base_url["name"])
            self.path_url.write_text(json.dumps(url_dict))


OWLITE_SETTINGS = OwLiteSettings()
