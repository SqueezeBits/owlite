"""Module for OwLite Authentication

Includes functions for handling OwLite user authentication.
"""

import re
from getpass import getpass

from requests import HTTPError

from ..logger import log
from ..owlite_settings import OWLITE_SETTINGS
from .login_api import login as _login
from .login_api import whoami


def login() -> None:
    """Login to OwLite.

    Raises:
        HTTPError: When login request was not successful.
    """

    def _check_email(email: str) -> bool:
        """Checks integrity of email using RegEx.

        Args:
            email (str): A email to check.

        Returns:
            bool: True if given email is valid, False otherwise.
        """
        regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        if not re.fullmatch(regex, email):
            log.error("Invalid ID(email) provided. Please enter email format")
            return False
        return True

    def _check_password(password: str) -> bool:
        r"""Check if given password fulfills password condition.

        Args:
            password (str): A password to check.

        Returns:
            bool: True if given password is valid, False otherwise.
        """
        allowed_specials = r"!@#$%^&*()_+\-=\[\]{}|~₩"
        regex = r"^(?=.*[!@#$%^&*()_+\-=\[\]{}|~₩])[A-Za-z0-9!@#$%^&*()_+\-=\[\]{}|~₩]{8,}$"
        if not re.match(regex, password):
            log.error(
                "Invalid password provided. "
                "Please enter Minimum eight characters, at least one letter, "
                f"one number and one special character among {allowed_specials}"
            )
            return False
        return True

    email = input("Enter your ID(e-mail): ")
    if not _check_email(email):
        return
    password = getpass("Enter your password: ")
    if not _check_password(password):
        return

    try:
        tokens = _login(email, password)
    except HTTPError:
        return
    OWLITE_SETTINGS.tokens = tokens

    username = whoami()
    log.info(f"Currently logged in as : {username}")
    log.info(f"Your authentication token has been saved to {OWLITE_SETTINGS.path_tokens}")


def logout() -> None:
    """Logout from OwLite, tokens are deleted from the machine."""
    tokens = OWLITE_SETTINGS.tokens
    if tokens is None:
        log.error("This device is not logged in")
        return

    OWLITE_SETTINGS.tokens = None
    log.info("Successfully logged out")
