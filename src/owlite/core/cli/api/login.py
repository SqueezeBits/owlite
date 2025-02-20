"""API wrapper module for login."""

import requests
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from ...api_base import APIBase
from ...cache.workspace import Workspace
from ...constants import OWLITE_API_DEFAULT_TIMEOUT
from ...exceptions import LoginError
from ...logger import log
from ...settings import OWLITE_SETTINGS


class UserInfo(BaseModel):
    """User Information."""

    model_config = ConfigDict(extra="ignore")
    name: str = Field(validation_alias=AliasChoices("name", "username"))
    default_workspace_id: str = Field(validation_alias=AliasChoices("workspace", "default_workspace_id"))
    priority_queues_count: int = Field(
        validation_alias=AliasChoices("priority_queues_count", "monthly_benchmark_count")
    )


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
                log.error(login_failed_dict[resp["detail"]])  # UX
            raise LoginError("Login failed")

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
        log.error("Please log in using 'owlite login'. Account not found on this device")  # UX
        raise LoginError("OwLite token not found")

    main_api = APIBase(OWLITE_SETTINGS.base_url.MAIN, "OWLITE_LOGIN_API")
    try:
        resp = main_api.post("/login/whoami")
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 403:
            raise LoginError("Not authenticated") from e
        raise e
    assert isinstance(resp, dict)
    log.debug(f"whoami response: {resp}")
    user_info = UserInfo.model_validate(resp)
    log.debug(f"user info : {user_info}")

    if (workspace := OWLITE_SETTINGS.current_workspace) is None:
        workspace_id = user_info.default_workspace_id
        log.warning("No workspace selected. Automatically selecting the default one")  # UX
    else:
        workspace_id = workspace.id
    OWLITE_SETTINGS.current_workspace = Workspace.load(workspace_id)

    return user_info
