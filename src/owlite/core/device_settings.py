from functools import cached_property
from typing import get_args

from ..enums.runtime import Runtime
from .api_base import NEST_API_BASE
from .cache import OWLITE_CACHE_PATH
from .cache.device import Device
from .cache.text import read_text, write_text
from .constants import SUPPORTED_QUALCOMM_DEVICES
from .exceptions import DeviceError, LoginError
from .logger import log
from .settings import OWLITE_SETTINGS


class OwLiteDeviceSettings:
    """Handles device settings and cache management for OwLite.

    OwLiteDeviceSettings manages device settings and cache information for OwLite.
    It provides methods to retrieve, add, and remove device managers, as well as
    to set and retrieve connected devices.

    Attributes:
        connected_cache (Path): Path to store information about the connected device.
    """

    def __init__(self) -> None:
        """Initialize OwLite device settings.

        Initialize paths for OwLite cache directory to store information about
        the connected device.
        """
        self.connected_cache = OWLITE_CACHE_PATH / "connected"

    @property
    def connected(self) -> Device | None:
        """Retrieve the connected device.

        Returns:
            Device | None: An instance representing the connected device, or None if no device is selected
        """
        connected = read_text(self.connected_cache)
        if connected is None:
            return None
        return Device.model_validate_json(connected)

    @connected.setter
    def connected(self, connected: Device | None = None) -> None:
        """Connect to the device manager and selects a device or deletes a device setting from storage.

        Does not fail if the device does not exist.

        Args:
            connected (Device | None): The instance representing the connected device
        """
        if connected:
            write_text(self.connected_cache, connected.model_dump_json())
        else:
            self.connected_cache.unlink(missing_ok=True)

    @cached_property
    def devices(self) -> dict[str, Device]:
        """The dictionary of devices managed by NEST."""
        tokens = OWLITE_SETTINGS.tokens
        if not tokens:
            log.error("Please login using 'owlite login' to connect to NEST devices")  # UX
            raise LoginError("Not authenticated")

        if (workspace := OWLITE_SETTINGS.current_workspace) is None:
            log.error("No workspace selected. Please select a workspace")  # UX
            raise RuntimeError("No workspace selected")

        try:
            resp = NEST_API_BASE.get(
                "/devices",
                params={"workspace_id": workspace.id},
            )
        except Exception as err:
            raise DeviceError(err) from err

        assert isinstance(resp, list)

        return {device["name"]: Device(**device) for device in resp}

    def get_device(self, name: str) -> Device:
        """Get the device by its name.

        Args:
            name (str): The name of a device to retrieve.

        Raises:
            RuntimeError: If no device with the provided name is found.

        Returns:
            Device: The device with the provided name.
        """
        assert self.connected
        if self.connected.runtime == Runtime.QNN:
            assert name in get_args(SUPPORTED_QUALCOMM_DEVICES)
            return Device(name=self.connected.name, runtime=self.connected.runtime, runtime_extra=name)

        if name not in self.devices:
            log.error(f"No such device: {name}. Available devices are {', '.join(self.devices.keys())}")  # UX
            raise RuntimeError("Device not found")
        return self.devices[name]


OWLITE_DEVICE_SETTINGS = OwLiteDeviceSettings()
