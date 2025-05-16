"""Handles device-related commands in OwLite CLI using argparse."""

# pylint: disable=unnecessary-lambda, too-few-public-methods
from argparse import Namespace, _SubParsersAction

from .. import BaseOwLiteCLICommand
from ..device import connect_device, disconnect_device


class DeviceCommands(BaseOwLiteCLICommand):
    """Handle device-related commands in OwLite CLI."""

    @staticmethod
    def register_subcommand(parser: _SubParsersAction) -> None:
        """Register subcommands for device-related operations.

        Args:
            parser (_SubParsersAction): The parser object to add subcommands to.
        """
        device_parser = parser.add_parser("device", help="Device setting from owlite")
        device_parser.add_argument(
            "mode",
            choices=["connect", "disconnect"],
            help="Device setting command",
        )

        device_parser.set_defaults(func=lambda args: DeviceCommand(args))


class DeviceCommand:
    """Handles device-specific commands in OwLite CLI."""

    def __init__(self, args: Namespace) -> None:
        """Initialize the DeviceCommand.

        Args:
            args: Arguments passed to the command.
        """
        self.args = args

    def run(self) -> None:
        """Execute the specified device-related operation based on the mode specified."""
        if self.args.mode == "connect":
            connect_device()
        elif self.args.mode == "disconnect":
            disconnect_device()
