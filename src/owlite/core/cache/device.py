from functools import cached_property
from typing import Any

import requests
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from ...enums.runtime import Runtime
from ..api_base import APIBase
from ..exceptions import DeviceError, LoginError
from ..logger import log
from ..settings import OWLITE_SETTINGS


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
        try:
            devices = self.devices
        except DeviceError as err:
            log.error(f"Failed to validate the device manager '{self.name}'")
            raise err
        log.debug(f"Devices at {self.name}({self.url})\n{devices}")

    @cached_property
    def devices(self) -> dict[str, "Device"]:
        """The devices managed by this device manager."""
        tokens = OWLITE_SETTINGS.tokens
        if not tokens and self.url == OWLITE_SETTINGS.base_url.NEST:
            log.error("Using OwLite default device manager needs login. Please login using 'owlite login'")  # UX
            raise LoginError("Not authenticated")

        if (workspace := OWLITE_SETTINGS.current_workspace) is None:
            log.error("No workspace selected. Please select a workspace")  # UX
            raise RuntimeError("No workspace selected")

        device_api_base = APIBase(self.url, "DEVICE_MANAGER_API")
        try:
            resp = device_api_base.get("/devices", params={"workspace_id": workspace.id})
        except requests.ConnectionError as err:
            log.error(
                f"Failed to establish a connection. Please check if the device manager url '{self.url}' is correct"
            )  # UX
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
            name (str): The name of a device to retrieve.

        Raises:
            RuntimeError: If no device with the provided name is found.

        Returns:
            Device: The device with the provided name.
        """
        if name not in self.devices:
            log.error(f"No such device: {name}. Available devices are {', '.join(self.devices.keys())}")  # UX
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
