"""API wrapper module for login"""

import requests

from ..constants import OWLITE_API_DEFAULT_TIMEOUT
from ..logger import log
from ..owlite_settings import OWLITE_SETTINGS


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
    main_url = OWLITE_SETTINGS.base_url["MAIN"]
    front_url = OWLITE_SETTINGS.base_url["FRONT"]
    payload = {"username": email, "password": password}

    response = requests.post(f"{main_url}/login", data=payload, timeout=OWLITE_API_DEFAULT_TIMEOUT)
    resp = response.json()

    if not response.ok:
        if response.status_code == 401:
            login_failed_dict = {
                "User not found": "Login failed with unknown user. Please check your ID(email) again "
                f"or visit {front_url}/auth/login and sign up",
                "Incorrect password": "Login failed with incorrect password. Please check your password again",
            }
            if resp and resp["detail"] in login_failed_dict:
                log.error(login_failed_dict[resp["detail"]])

        response.raise_for_status()

    assert isinstance(resp, dict)
    return resp


def whoami() -> str:
    """Get username with current access token at owlite cache.

    Raises:
        RuntimeError: When no saved login token found.
        HTTPError: when request was not successful.

    Returns:
        str: Username.
    """
    main_url = OWLITE_SETTINGS.base_url["MAIN"]

    tokens = OWLITE_SETTINGS.tokens
    if not tokens:
        log.error("This device is not authenticated. Please log in using 'owlite login'")
        raise RuntimeError("Not authenticated")

    def _whoami(access_token: str) -> requests.Response:
        return requests.post(
            f"{main_url}/login/whoami",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=OWLITE_API_DEFAULT_TIMEOUT,
        )

    response = _whoami(tokens["access_token"])
    if not response.ok:
        if response.status_code == 401:
            refresh_res = requests.post(
                f"{main_url}/login/refresh",
                json={"refresh_token": tokens["refresh_token"]},
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
                timeout=OWLITE_API_DEFAULT_TIMEOUT,
            )
            if not refresh_res.ok:
                log.error("Login session expired, please log in again by 'owlite login'")
                OWLITE_SETTINGS.tokens = None
                refresh_res.raise_for_status()

            refresh_dict = refresh_res.json()
            assert isinstance(refresh_dict, dict)
            OWLITE_SETTINGS.tokens = refresh_dict

            response = _whoami(refresh_dict["access_token"])
        else:
            log.error("Invalid token used, please log in again using 'owlite login'")
            OWLITE_SETTINGS.tokens = None
            response.raise_for_status()

    resp = response.json()
    assert isinstance(resp, dict)
    username = resp["username"]
    return str(username)
