import json

from .cache import OWLITE_CACHE_PATH
from .cache.device import Device, DeviceManager
from .cache.text import read_text, write_text
from .owlite_settings import OWLITE_SETTINGS


class OwLiteDeviceSettings:
    """Handles device settings and cache management for OwLite.

    OwLiteDeviceSettings manages device settings and cache information for OwLite.
    It provides methods to retrieve, add, and remove device managers, as well as
    to set and retrieve connected devices.

    Attributes:
        devices_cache (Path): Path to store device manager information.
        connected_cache (Path): Path to store information about the connected device.
    """

    def __init__(self) -> None:
        """Initialize OwLite device settings.

        Initialize paths for OwLite cache directory to store device manager information
        and information about the connected device.
        """
        self.devices_cache = OWLITE_CACHE_PATH / "devices"
        self.connected_cache = OWLITE_CACHE_PATH / "connected"

    @property
    def managers(self) -> dict[str, "DeviceManager"]:
        """Retrieve the device manager dictionary.

        Returns:
            dict[str, DeviceManager]: Device manager dictionary
        """
        default_manager = DeviceManager(name="NEST", url=OWLITE_SETTINGS.base_url.NEST)
        registered_managers = {"NEST": default_manager}

        cache_content = read_text(self.devices_cache)
        if cache_content is None:
            return registered_managers

        cached_managers: dict[str, DeviceManager] = {
            key: DeviceManager.model_validate_json(val) for key, val in json.loads(cache_content).items()
        }
        assert cached_managers
        registered_managers.update(cached_managers)
        return registered_managers

    @managers.setter
    def managers(self, managers: dict[str, "DeviceManager"]) -> None:
        managers.pop("NEST")
        device_dict_json = json.dumps(managers, default=lambda o: o.model_dump_json())
        write_text(self.devices_cache, device_dict_json)

    def add_manager(self, manager: "DeviceManager") -> None:
        """Add a new device to the cache.

        Args:
            manager (DeviceManager): a new manager
        """
        managers = self.managers
        managers[manager.name] = manager
        self.managers = managers

    def remove_manager(self, name: str) -> None:
        """Remove an existing device manager from the cache.

        Args:
            name (str): the name of the manager to remove
        """
        managers = self.managers
        managers.pop(name, None)
        managers.pop("NEST")
        if bool(managers):
            self.managers = managers
        else:
            self.devices_cache.unlink(missing_ok=True)

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


OWLITE_DEVICE_SETTINGS = OwLiteDeviceSettings()
