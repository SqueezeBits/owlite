"""Device Management Module

Provides functions to manage device connections, device managers, and settings."""

from .. import Device, DeviceManager
from ..exceptions import DeviceError
from ..logger import log
from ..owlite_settings import OWLITE_SETTINGS
from .api.device import get_devices


def get_device_list(url: str) -> list[str]:
    """Get devices connected to current device manager.

    Raises:
        HTTPError: When request was not successful.

    Returns:
        list[str]: A list of device names connected to device manager.
    """

    device_list = get_devices(url)
    device_names: list[str] = []
    for device in device_list:
        assert isinstance(device, dict)
        device_names.append(str(device.get("name", "")))

    devices_repr = "\n".join(device_names)
    log.debug(f"Found following devices from device manager : {url}\n{devices_repr}")

    return device_names


def add_manager(name: str, url: str) -> None:
    """Adds a new device manager with the given name and url.

    Args:
        name (str): The name of the device manager.
        url (str): The url of the device manager.

    Raises:
        DeviceError: If the device manager name already exists.
        HTTPError: If the request to the url fails.
    """
    if name in OWLITE_SETTINGS.managers:
        log.error(f"The device '{name}' already exists. Please give it a different name")
        raise DeviceError(f"Duplicate device found: {name}")

    get_devices(url)

    OWLITE_SETTINGS.add_manager(DeviceManager(name, url))
    print_manager_list()


def print_manager_list() -> None:
    """Prints a list of added device managers"""
    connected_device = OWLITE_SETTINGS.connected_device
    device_list = "\n".join(
        f"{manager.name}: {manager.url}"
        + (
            f" -> connected to device '{connected_device.name}'"
            if connected_device and manager.name == connected_device.manager.name
            else ""
        )
        for manager in OWLITE_SETTINGS.managers.values()
    )
    log.info(f"Available device managers:\n{device_list}")


def remove_manager(name: str) -> None:
    """Removes the specified device manager.

    Args:
        name (str): The name of the device manager to remove.

    Raises:
        DeviceError: If the specified device manager does not exist or is NEST.
    """
    if name == "NEST":
        log.error(f"Unable to delete the device as the device named '{name}'")
        raise DeviceError(f"Invalid device name : {name}")
    if name not in OWLITE_SETTINGS.managers:
        log.error(f"Unable to delete the device as the device named '{name}' does not exist")
        raise DeviceError(f"Invalid device name : {name}")
    OWLITE_SETTINGS.remove_manager(name)
    log.info(f"Removed the device manager: {name}")

    connected_device = OWLITE_SETTINGS.connected_device
    if connected_device and name == connected_device.manager.name:
        disconnect_device()
    print_manager_list()


def connect_device(name: str) -> None:
    """Connects to the device in selected device manager.

    Raises:
        DeviceError: If the selected device manager doesn't exist.
    """
    if name not in OWLITE_SETTINGS.managers:
        log.error(f"No such device: '{name}'. Please add a device using 'owlite device add --name (name) --url (url)'")
        raise DeviceError("Device not found")
    manager = OWLITE_SETTINGS.managers[name]
    assert manager.url
    device = _select_device(manager.url)
    OWLITE_SETTINGS.connected_device = Device(device, manager)
    log.info(f"Connected to the device '{device}' at '{manager.name}' ({manager.url})")


def _select_device(url: str) -> str:
    """Internal function to connect to the device and prompt the user to choose.

    Raises:
        ValueError: If an invalid index or device is chosen.

    Returns:
        str: The name of the connected device.
    """

    device_list = get_device_list(url)
    _device_list = "\n".join(f"{index}: {name}" for index, name in enumerate(device_list))
    log.info(f"Available devices:\n{_device_list}")

    user_input = input("Enter the index of the device you want to connect to: ")
    try:
        index = int(user_input)
        if index not in range(len(device_list)):
            log.error(
                f"Index out of range. Please choose the device index within the range [0, {len(device_list) - 1}]"
            )
            raise DeviceError(f"Invalid index given : {index}")
        device = device_list[index]
    except ValueError as e:
        log.error(f"Please provide a valid index within the range [0, {len(device_list) - 1}]")
        raise DeviceError(e) from e

    return device


def disconnect_device() -> None:
    """Disconnects the currently connected device, if any"""
    connected_device = OWLITE_SETTINGS.connected_device
    if connected_device is None:
        log.warning("No connected device found")
        return
    OWLITE_SETTINGS.connected_device = None
    log.info(f"Disconnected from the device '{connected_device.name}'")


CONNECTED_DEVICE = OWLITE_SETTINGS.connected_device
OWLITE_DEVICE_NAME = CONNECTED_DEVICE.name if CONNECTED_DEVICE else None
