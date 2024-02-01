import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from .constants import (
    NEST_URL,
    OWLITE_DOVE_API_BASE_URL,
    OWLITE_FRONT_BASE_URL,
    OWLITE_MAIN_API_BASE_URL,
)

# default cache
owlite_cache_dir = os.path.expanduser(
    os.getenv(
        "OWLITE_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "owlite"),
    )
)


def read_text(path: Path) -> Optional[str]:
    """Reads text from a file specified by the given path.

    Args:
        path (Path): The path to the file.

    Returns:
        Optional[str]: Text read from the file or None if the file is not found.
    """
    try:
        text = path.read_text(encoding="utf-8")
        text = text.replace("\r", "").replace("\n", "").strip()
        return text
    except FileNotFoundError:
        return None


@dataclass
class Tokens:
    """Represents tokens.

    Attributes:
        access_token (str): access token for OwLite login.
        refresh_token (str): refresh token for OwLite login.
    """

    access_token: str
    refresh_token: str


@dataclass
class DeviceManager:
    """Represents a device manager with a name and an optional URL.

    Attributes:
        name (str): The name of the device manager.
        url (Optional[str]): The URL of the device manager.
    """

    name: str
    url: Optional[str] = None


@dataclass
class Device:
    """Represents a device with device manager which connected from.

    Attributes:
        name (str): The name of the device.
        manager (DeviceManager): The device manager which has device with name.
    """

    name: str
    manager: DeviceManager


# pylint:disable=too-few-public-methods
class BaseURLs:
    """Represents base urls.

    Attributes:
        FRONT (str): The url for OwLite front server.
        MAIN (str): The url for OwLite main server.
        DOVE (str): The url for OwLite Dove server.
    """

    _default_urls = {
        "FRONT": OWLITE_FRONT_BASE_URL,
        "MAIN": OWLITE_MAIN_API_BASE_URL,
        "DOVE": OWLITE_DOVE_API_BASE_URL,
        "NEST": NEST_URL,
    }

    def __init__(
        self,
        FRONT: str = _default_urls["FRONT"],  # noqa: N803
        MAIN: str = _default_urls["MAIN"],  # noqa: N803
        DOVE: str = _default_urls["DOVE"],  # noqa: N803
        NEST: str = _default_urls["NEST"],  # noqa: N803
    ) -> None:
        # pylint: disable=invalid-name
        self.FRONT = FRONT
        self.MAIN = MAIN
        self.DOVE = DOVE
        self.NEST = NEST
        # pylint: enable=invalid-name

    def set(self, name: str, url: Optional[str] = None) -> None:
        """Sets the given URL to input value or its default value.

        Args:
            name (str): The name of the URL to reset.
        """
        if name in self._default_urls:
            if url:
                setattr(self, name, url)
            else:
                setattr(self, name, self._default_urls[name])
        else:
            raise ValueError(f"Invalid name: {name}")


class ClassEncoder(json.JSONEncoder):
    """Encodes `DeviceManager` and `Device` objects to JSON."""

    def default(self, o: Any) -> dict[str, Any]:
        """
        Overrides the default JSON serialization for `Tokens`, `DeviceManager`, `Device` and `BaseURLs`.

        Args:
            obj: The object to be serialized.

        Returns:
            A dictionary representation of the object.
        """
        if isinstance(o, Tokens):
            return {"access_token": o.access_token, "refresh_token": o.refresh_token}
        if isinstance(o, DeviceManager):
            return {"name": o.name, "url": o.url}
        if isinstance(o, Device):
            return {"name": o.name, "manager": {"name": o.manager.name, "url": o.manager.url}}
        if isinstance(o, BaseURLs):
            return {"FRONT": o.FRONT, "MAIN": o.MAIN, "DOVE": o.DOVE, "NEST": o.NEST}
        return super().default(o)


class ClassDecoder(json.JSONDecoder):
    """Decodes JSON data to `Tokens`, `DeviceManager`, `Device` and `BaseURLs` objects."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the `ClassDecoder` with a custom object hook.

        Args:
            *args: Additional arguments for `json.JSONDecoder`.
            **kwargs: Additional keyword arguments for `json.JSONDecoder`.
        """
        super().__init__(*args, object_hook=self.dict_to_class, **kwargs)

    def dict_to_class(self, dct: dict[str, Any]) -> Union[Tokens, DeviceManager, Device, BaseURLs, dict[str, Any]]:
        """Converts a dictionary representation to a `DeviceManager`, `Device` and `BaseURLs` object.

        Args:
            dct: A dictionary containing specific keys.

        Returns:
            A `Tokens`, `DeviceManager`, `Device` and `BaseURLs` object or the original dictionary if invalid.
        """
        if "access_token" in dct:
            return Tokens(access_token=dct["access_token"], refresh_token=dct["refresh_token"])
        if "FRONT" in dct:
            if "NEST" not in dct:
                return BaseURLs(FRONT=dct["FRONT"], MAIN=dct["MAIN"], DOVE=dct["DOVE"], NEST=NEST_URL)
            return BaseURLs(FRONT=dct["FRONT"], MAIN=dct["MAIN"], DOVE=dct["DOVE"], NEST=dct["NEST"])
        if "name" not in dct:
            return dct
        if "manager" in dct:
            assert isinstance(dct["manager"], DeviceManager)
            return Device(name=dct["name"], manager=dct["manager"])
        return DeviceManager(name=dct["name"], url=dct.get("url"))
