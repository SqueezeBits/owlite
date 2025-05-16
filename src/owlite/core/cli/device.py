"""Device Management Module.

Provides functions to manage device connections.
"""

from ..cache.device import Device
from ..device_settings import OWLITE_DEVICE_SETTINGS
from ..exceptions import DeviceError
from ..logger import log


def connect_device() -> None:
    """Connect to the device in selected device manager.

    Raises:
        DeviceError: If the selected device manager doesn't exist.
    """
    devices = list(OWLITE_DEVICE_SETTINGS.devices.values())
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
    except ValueError as e:
        log.error(f"Please provide a valid index within the range [0, {len(devices) - 1}]")  # UX
        raise DeviceError(e) from e

    OWLITE_DEVICE_SETTINGS.connected = device

    log.info(f"Connected to the device '{OWLITE_DEVICE_SETTINGS.connected}'")


def disconnect_device() -> None:
    """Disconnect the currently connected device, if any."""
    OWLITE_DEVICE_SETTINGS.connected = None
    log.info("Successfully disconnected from the device")  # UX


def connect_to_first_available_device() -> Device:
    """Connect to the free device in NEST.

    Returns:
        Device: connected device name
    """
    if OWLITE_DEVICE_SETTINGS.connected:
        return OWLITE_DEVICE_SETTINGS.connected
    log.info("Connecting to the first device at NEST")  # UX
    device = list(OWLITE_DEVICE_SETTINGS.devices.values())[0]
    OWLITE_DEVICE_SETTINGS.connected = device
    return device
