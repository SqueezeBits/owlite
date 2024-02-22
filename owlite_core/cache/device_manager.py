from typing import Optional

from pydantic import BaseModel, Field


class DeviceManager(BaseModel):
    """Represents a device manager with a name and an optional URL.

    Attributes:
        name (str): The name of the device manager.
        url (Optional[str]): The URL of the device manager.
    """

    name: str
    url: Optional[str] = Field(default=None)


class Device(BaseModel):
    """Represents a device with device manager which connected from.

    Attributes:
        name (str): The name of the device.
        manager (DeviceManager): The device manager which has device with name.
    """

    name: str
    manager: DeviceManager
