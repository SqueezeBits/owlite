"""Handles user-related commands in OwLite CLI."""

# pylint: disable=unnecessary-lambda, too-few-public-methods
from argparse import Namespace, _SubParsersAction

from ...logger import log
from ...settings import OWLITE_SETTINGS
from .. import BaseOwLiteCLICommand
from ..api.login import whoami
from ..login import login, logout


class UserCommands(BaseOwLiteCLICommand):
    """Handles user-related commands in OwLite CLI."""

    @staticmethod
    def register_subcommand(parser: _SubParsersAction) -> None:
        """Register subcommands for user-related operations.

        Args:
            parser (_SubParsersAction): The parser object to add subcommands to.
        """
        login_parser = parser.add_parser("login", help="Login to OwLite")
        login_parser.set_defaults(func=lambda args: LoginCommand(args))
        whoami_parser = parser.add_parser("whoami", help="Display the current user's username")
        whoami_parser.set_defaults(func=lambda args: WhoamiCommand(args))
        logout_parser = parser.add_parser("logout", help="Log out")
        logout_parser.set_defaults(func=lambda args: LogoutCommand(args))


class BaseUserCommand:
    """Base class for user-related commands."""

    def __init__(self, args: Namespace) -> None:
        """Initialize the BaseUserCommand.

        Args:
            args: Arguments passed to the command.
        """
        self.args = args


class LoginCommand(BaseUserCommand):
    """Handle the 'login' command."""

    def run(self) -> None:
        """Execute the login operation."""
        login()
        if OWLITE_SETTINGS.current_workspace is not None:
            log.info(f"Your Current Workspace: {OWLITE_SETTINGS.current_workspace.name}")  # UX
            log.info("The OwLite Package operates within the selected workspace.")  # UX


class WhoamiCommand(BaseUserCommand):
    """Handle the 'whoami' command."""

    def run(self) -> None:
        """Execute the whoami operation and prints the username."""
        userinfo = whoami()
        log.info(userinfo.name)  # UX
        if OWLITE_SETTINGS.current_workspace is not None:
            log.info(f"Your Current Workspace: {OWLITE_SETTINGS.current_workspace.name}")  # UX
            log.info("The OwLite Package operates within the selected workspace.")  # UX


class LogoutCommand(BaseUserCommand):
    """Handle the 'logout' command."""

    def run(self) -> None:
        """Execute the logout operation."""
        logout()
