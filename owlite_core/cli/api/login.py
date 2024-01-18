"""API wrapper module for login"""

from dataclasses import dataclass

import requests

from ...api_base import APIBase
from ...constants import OWLITE_API_DEFAULT_TIMEOUT
from ...exceptions import LoginError
from ...logger import log
from ...owlite_settings import OWLITE_SETTINGS


@dataclass
class UserInfo:
    """User Information"""

    name: str
    tier: int


def login(email: str, password: str) -> dict[str, str]:
    """Attempt login with given email and password, returns dict of tokens if login was successful.

    Args:
        email (str): Email.
        password (str): Password.

    Raises:
        HTTPError: When login was not successful.

    Returns:
        dict[str, str]: A dictionary containing access token and refresh token.
    """
    main_url = OWLITE_SETTINGS.base_url.MAIN
    front_url = OWLITE_SETTINGS.base_url.FRONT
    payload = {"username": email, "password": password}

    response = requests.post(f"{main_url}/login", data=payload, timeout=OWLITE_API_DEFAULT_TIMEOUT)
    resp = response.json()

    if not response.ok:
        if response.status_code == 401:
            login_failed_dict = {
                "User not found": (
                    "The email is not registered. Please check if your email is correct "
                    f"or sign up at {front_url}/auth/login"
                ),
                "Incorrect password": "Incorrect password provided. Please check if your password is correct",
            }
            if resp and resp["detail"] in login_failed_dict:
                log.error(login_failed_dict[resp["detail"]])
            raise LoginError

        response.raise_for_status()

    assert isinstance(resp, dict)
    return resp


def whoami() -> UserInfo:
    """Get username with current access token at owlite cache.

    Raises:
        LoginError: When no saved login token found.
        HTTPError: when request was not successful.

    Returns:
        UserInfo: Information of current user.
    """
    if OWLITE_SETTINGS.tokens is None:
        log.error("Please log in using 'owlite login'. Account not found on this device")
        raise LoginError("OwLite token not found")

    main_api = APIBase(OWLITE_SETTINGS.base_url.MAIN, "OWLITE_LOGIN_API")
    resp = main_api.post("/login/whoami")
    assert isinstance(resp, dict)
    current_user = UserInfo(name=str(resp["username"]), tier=int(resp["tier"]))
    log.debug(f"user info: {current_user.name}, {current_user.tier}")
    return current_user
