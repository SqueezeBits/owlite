"""Module for OwLite Authentication

Includes functions for handling OwLite user authentication.
"""

import re
from getpass import getpass

from ..cache.tokens import Tokens
from ..logger import log
from ..owlite_settings import OWLITE_SETTINGS
from .api.login import login as _login
from .api.login import whoami


def login() -> None:
    """Login to OwLite.

    Raises:
        HTTPError: When login request was not successful.
    """

    def _is_valid_email(email: str) -> bool:
        """Checks if the email is valid.

        Args:
            email (str): A email to check.

        Returns:
            bool: True if given email is valid, False otherwise.
        """
        regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        if not re.fullmatch(regex, email):
            log.error("Invalid email provided")  # UX
            return False
        return True

    def _is_valid_password(password: str) -> bool:
        r"""Check if the password is valid.

        Args:
            password (str): A password to check.

        Returns:
            bool: True if given password is valid, False otherwise.
        """
        allowed_specials = r"!@#$%^&*()_+\-=\[\]{}|~₩"
        regex = r"^(?=.*[!@#$%^&*()_+\-=\[\]{}|~₩])[A-Za-z0-9!@#$%^&*()_+\-=\[\]{}|~₩]{8,}$"
        if not re.match(regex, password):
            log.error(
                "The password does not meet the requirement. A valid password must contain at least eight characters, "
                "including one or more alphabetic, numeric, and special characters. "
                f"Special characters must be chosen from {allowed_specials}"
            )  # UX
            return False
        return True

    email = input("Enter your email: ")  # UX
    if not _is_valid_email(email):
        return
    password = getpass("Enter your password: ")  # UX
    if not _is_valid_password(password):
        return

    resp = _login(email, password)
    tokens = Tokens(access_token=resp["access_token"], refresh_token=resp["refresh_token"])
    OWLITE_SETTINGS.tokens = tokens

    userinfo = whoami()
    log.info(f"Logged in as {userinfo.name}")  # UX
    log.info(f"Your price plan: {userinfo.plan.name}")  # UX
    log.info(f"Your workgroup: {userinfo.workgroup}")  # UX
    log.info(f"Your authentication token is saved at {OWLITE_SETTINGS.tokens_cache}")  # UX
    log.debug(f"Saved tokens: \n\t\taccess token= '{tokens.access_token}'\n\t\trefresh token= '{tokens.refresh_token}'")


def logout() -> None:
    """Logout from OwLite, tokens are deleted from the machine."""
    OWLITE_SETTINGS.tokens = None
    log.info("Successfully logged out")  # UX
