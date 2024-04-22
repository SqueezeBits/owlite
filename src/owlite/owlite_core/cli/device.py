"""Device Management Module.

Provides functions to manage device connections, device managers, and settings.
"""


from ..cache.device import Device, DeviceManager
from ..exceptions import DeviceError
from ..logger import log
from ..owlite_device_settings import OWLITE_DEVICE_SETTINGS


def add_manager(name: str, url: str) -> None:
    """Add a new device manager with the given name and url.

    Args:
        name (str): The name of the device manager.
        url (str): The url of the device manager.

    Raises:
        DeviceError: If the device manager name already exists.
        HTTPError: If the request to the url fails.
    """
    if name in OWLITE_DEVICE_SETTINGS.managers:
        log.error(f"The device '{name}' already exists. Please give it a different name")  # UX
        raise DeviceError(f"Duplicate device manager found: {name}")

    manager = DeviceManager(name=name, url=url)

    OWLITE_DEVICE_SETTINGS.add_manager(manager)
    connected = OWLITE_DEVICE_SETTINGS.connected
    if connected and name == connected.manager.name and url != connected.manager.url:
        disconnect_device()
    print_manager_list()


def print_manager_list() -> None:
    """Print a list of added device managers."""
    connected = OWLITE_DEVICE_SETTINGS.connected
    device_list = "\n".join(
        f"{manager.name}: {manager.url}"
        + (
            f" -> connected to device '{connected.name}'"
            if connected and manager.name == connected.manager.name
            else ""
        )
        for manager in OWLITE_DEVICE_SETTINGS.managers.values()
    )
    log.info(f"Available device managers:\n{device_list}")  # UX


def remove_manager(name: str) -> None:
    """Remove the specified device manager.

    Args:
        name (str): The name of the device manager to remove.

    Raises:
        DeviceError: If the specified device manager does not exist or is NEST.
    """
    if name == "NEST":
        log.error("Unable to delete the NEST device manager")  # UX
        raise DeviceError(f"Invalid device name : {name}")
    if name not in OWLITE_DEVICE_SETTINGS.managers:
        log.error(f"Unable to delete the device manager '{name}' does not exist")  # UX
        raise DeviceError(f"Invalid device name : {name}")
    OWLITE_DEVICE_SETTINGS.remove_manager(name)
    log.info(f"Removed the device manager: {name}")  # UX

    connected = OWLITE_DEVICE_SETTINGS.connected
    if connected and name == connected.manager.name:
        disconnect_device()
    print_manager_list()


def connect_device(name: str) -> None:
    """Connect to the device in selected device manager.

    Raises:
        DeviceError: If the selected device manager doesn't exist.
    """
    if name not in OWLITE_DEVICE_SETTINGS.managers:
        log.error(
            f"No such device: '{name}'. Please add a device using 'owlite device add --name (name) --url (url)'"
        )  # UX
        raise DeviceError("Device not found")
    manager = OWLITE_DEVICE_SETTINGS.managers[name]
    OWLITE_DEVICE_SETTINGS.connected = _select_device(manager)
    log.info(f"Connected to the device '{OWLITE_DEVICE_SETTINGS.connected}' at '{manager}'")


def _select_device(manager: DeviceManager) -> Device:
    """Connect to the device and prompt the user to choose.

    Args:
        manager (DeviceManager): The device manager to connect.

    Raises:
        DeviceError: If an invalid index or device is chosen.

    Returns:
        Device: The device to connect.
    """
    devices = list(manager.devices.values())
    _devices = "\n".join(f"{index}: {device}" for index, device in enumerate(devices))
    log.info(f"Available devices:\n{_devices}")

    user_input = input("Enter the index of the device you want to connect to: ")  # UX
    try:
        index = int(user_input)
        if index not in range(len(devices)):
            log.error(
                f"Index out of range. Please choose the device index within the range [0, {len(devices) - 1}]"
            )  # UX
            raise DeviceError(f"Invalid index given : {index}")
        device = devices[index]
        return device
    except ValueError as e:
        log.error(f"Please provide a valid index within the range [0, {len(devices) - 1}]")  # UX
        raise DeviceError(e) from e


def disconnect_device() -> None:
    """Disconnect the currently connected device, if any."""
    connected = OWLITE_DEVICE_SETTINGS.connected
    if connected is None:
        log.warning("No connected device found")  # UX
        return
    OWLITE_DEVICE_SETTINGS.connected = None
    log.info(f"Disconnected from the device '{connected}'")  # UX


def connect_to_first_available_device() -> Device:
    """Connect to the free device in NEST device manager.

    Returns:
        Device: connected device name
    """
    if OWLITE_DEVICE_SETTINGS.connected:
        return OWLITE_DEVICE_SETTINGS.connected
    log.info("Connecting to the first device at NEST")  # UX
    manager = OWLITE_DEVICE_SETTINGS.managers["NEST"]
    device = list(manager.devices.values())[0]
    OWLITE_DEVICE_SETTINGS.connected = device
    return device
