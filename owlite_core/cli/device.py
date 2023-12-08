"""Device Management Module

Provides functions to manage device connections, device managers, and settings."""

import requests

from ..constants import OWLITE_API_DEFAULT_TIMEOUT
from ..logger import log
from ..owlite_settings import OWLITE_SETTINGS


def _get_devices(url: str) -> list[str]:
    """Get connected devices list from device manager.

    Args:
        url (str) : The url of a device manager.

    Raises:
        HTTPError: When request was not successful.

    Returns:
        list[str]: A list of device names connected to device manager.
    """
    try:
        response = requests.get(f"{url}/devices", timeout=OWLITE_API_DEFAULT_TIMEOUT)
    except requests.ConnectionError as err:
        log.error("Failed to establish a connection. Please verify the correctness of the device manager url")
        raise err
    except requests.exceptions.MissingSchema as err:
        log.error(
            "Invalid URL format. Please ensure a valid scheme (e.g., https://) "
            "is included in the device manager's url"
        )
        raise err
    if not response.ok:
        log.error(f"Received an error from request({url}/devices):\n{response.content.decode('utf-8')}")
        response.raise_for_status()

    resp = response.json()
    assert isinstance(resp, list)
    return resp


def get_device_list(url: str) -> list[str]:
    """Get devices connected to current device manager.

    Raises:
        HTTPError: When request was not successful.

    Returns:
        list[str]: A list of device names connected to device manager.
    """

    device_list = _get_devices(url)
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
        ValueError: If the device manager name already exists.
        HTTPError: If the request to the url fails.
    """
    manager_dict = OWLITE_SETTINGS.managers
    if manager_dict and name in manager_dict:
        log.error(f"The device '{name}' already exists. Please change the name and try again")
        raise ValueError(f"Invalid device name : {name}")

    _get_devices(url)

    OWLITE_SETTINGS.managers = {"name": name, "url": url}
    print_manager_list()


def print_manager_list() -> None:
    """Prints a list of added device managers"""
    manager_dict = OWLITE_SETTINGS.managers
    connected_device = OWLITE_SETTINGS.connected_device
    log.info("Current devices...")
    for manager, url in manager_dict.items():
        print(f"{manager} : {url}", end="")
        if connected_device and manager == connected_device["name"]:
            print(f" - connected with device: {connected_device['device']}")
        else:
            print("")


def remove_manager(name: str) -> None:
    """Removes the specified device manager.

    Args:
        name (str): The name of the device manager to remove.

    Raises:
        KeyError: If the specified device manager does not exist or is DEFAULT.
    """
    if name == "DEFAULT":
        log.error(f"Unable to delete the device as the device named '{name}'")
        raise KeyError(f"Invalid device name : {name}")
    manager_dict = OWLITE_SETTINGS.managers
    if name not in manager_dict:
        log.error(f"Unable to delete the device as the device named '{name}' does not exist")
        raise KeyError(f"Invalid device name : {name}")
    OWLITE_SETTINGS.managers = {"name": name}
    log.info(f"Removed the device manager: {name}")

    connected_device = OWLITE_SETTINGS.connected_device
    if connected_device and name == connected_device["name"]:
        disconnect_device()
    print_manager_list()


def connect_device(name: str) -> None:
    """Connects to the device in selected device manager.

    Raises:
        ValueError: If the selected device manager doesn't exist.
    """
    manager_dict = OWLITE_SETTINGS.managers
    if name not in manager_dict:
        log.error(f"Device with name {name} does not exists. Please try 'owlite device add --name (name) --url (url)")
        raise ValueError("Device not found")
    log.info(f"Connects to device manager '{name}[{manager_dict[name]}]")
    device = _select_device(manager_dict[name])
    OWLITE_SETTINGS.connected_device = {"name": name, "url": manager_dict[name], "device": device}
    log.info(f"Connected to device: {device}")


def _select_device(url: str) -> str:
    """Internal function to connect to the device and prompt the user to choose.

    Raises:
        ValueError: If an invalid index or device is chosen.

    Returns:
        str: The name of the connected device.
    """

    device_list: list[str] = get_device_list(url)

    message = "Device list to select...\n"
    for index, name in enumerate(device_list):
        message = message + f"{index}: {name}\n"
    log.info(message.rstrip("\n"))

    user_input = input("Enter the index of the device you want to select: ")
    try:
        index = int(user_input)
        if index not in range(len(device_list)):
            log.error(
                f"Entered index not found. Please choose the device with index range [0, {len(device_list) - 1}])"
            )
            raise ValueError(f"Invalid index given : {index}")
        device = device_list[index]
    except ValueError as e:
        log.error("Invalid input entered. Please enter a valid numeric index to select the device")
        raise e

    return device


def disconnect_device() -> None:
    """Disconnects the currently connected device, if any"""
    connected_device = OWLITE_SETTINGS.connected_device
    if connected_device is None:
        log.warning("Disconnect cannot be processed as no connected device was found")
        return
    OWLITE_SETTINGS.connected_device = None
    log.info(f"Successfully disconnected from the device {connected_device['device']}")


CONNECTED_DEVICE = OWLITE_SETTINGS.connected_device
OWLITE_DEVICE_NAME = CONNECTED_DEVICE["device"] if CONNECTED_DEVICE else None
