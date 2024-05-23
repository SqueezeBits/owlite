from functools import cached_property
from typing import Any

import requests
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from ...enums.runtime import Runtime
from ..api_base import APIBase
from ..exceptions import DeviceError, LoginError
from ..logger import log
from ..owlite_settings import OWLITE_SETTINGS


class DeviceManager(BaseModel):
    """Represents a device manager with a name and a URL.

    Attributes:
        name (str): The name of the device manager.
        url (str): The URL of the device manager.
    """

    name: str
    url: str

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        devices = self.devices
        log.debug(f"Devices at {self.name}({self.url})\n{devices}")

    @cached_property
    def devices(self) -> dict[str, "Device"]:
        """The devices managed by this device manager."""
        tokens = OWLITE_SETTINGS.tokens
        if not tokens and self.url == OWLITE_SETTINGS.base_url.NEST:
            log.error("Using OwLite default device manager needs login. Please log in using 'owlite login'")  # UX
            raise LoginError("Not authenticated")

        device_api_base = APIBase(self.url, "DEVICE_MANAGER_API")
        try:
            resp = device_api_base.get("/devices")
        except requests.ConnectionError as err:
            log.error("Failed to establish a connection. Please if the device manager url is correct")  # UX
            raise DeviceError(err) from err
        except requests.exceptions.MissingSchema as err:
            log.error(
                "Invalid URL format. Please ensure a valid scheme (e.g., https://) "
                "is included in the device manager's url"
            )  # UX
            raise DeviceError(err) from err

        assert isinstance(resp, list)
        return {device["name"]: Device(manager=self, **device) for device in resp}

    def get(self, name: str) -> "Device":
        """Get the device by its name.

        Args:
            manager (DeviceManager): An instance representing the device manager.
            name (str): The name of the device to retrieve.

        Raises:
            RuntimeError: If no device matching the name is found.

        Returns:
            Device: The device that matches the name.
        """
        if name not in self.devices:
            log.error(
                f"The device '{name}' entered cannot be found in the connected device manager. "
                f"List of connected devices: {list(self.devices.keys())}"
            )  # UX
            raise RuntimeError("Device not found")
        return self.devices[name]

    def __str__(self) -> str:
        return f"{self.name} [{self.url}]"


class Device(BaseModel):
    """Represents a device.

    Attributes:
        name (str): The name of the device.
        manager (DeviceManager): The device manager of the device.
        runtime (Runtime): The runtime associated with the device.
    """

    model_config = ConfigDict(extra="ignore")
    name: str = Field(validation_alias=AliasChoices("name", "device_name"))
    manager: DeviceManager
    runtime: Runtime = Field(default=Runtime.TensorRT, validation_alias=AliasChoices("framework", "runtime"))

    def __str__(self) -> str:
        return f"{self.name} [{self.runtime.name}]"  # pylint: disable=no-member
