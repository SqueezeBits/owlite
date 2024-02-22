import requests

from ...api_base import APIBase
from ...exceptions import DeviceError, LoginError
from ...logger import log
from ...owlite_settings import OWLITE_SETTINGS


def get_devices(url: str) -> list:
    """Get connected devices list from device manager.

    Args:
        url (str) : The url of a device manager.

    Raises:
        HTTPError: When request was not successful.

    Returns:
        list: A list includes device names connected to device manager.
    """
    tokens = OWLITE_SETTINGS.tokens
    if not tokens and url == OWLITE_SETTINGS.base_url.NEST:
        log.error("Using OwLite default device manager needs login. Please log in using 'owlite login'")  # UX
        raise LoginError("Not authenticated")

    device_api_base = APIBase(url, "DEVICE_MANAGER_API")
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
    return resp
