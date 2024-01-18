"""Handles device-related commands in OwLite CLI using argparse"""
# pylint: disable=unnecessary-lambda, too-few-public-methods
from argparse import Namespace, _SubParsersAction

from .. import BaseOwLiteCLICommand
from ..device import (
    add_manager,
    connect_device,
    disconnect_device,
    print_manager_list,
    remove_manager,
)


class DeviceCommands(BaseOwLiteCLICommand):
    """Handles device-related commands in OwLite CLI."""

    @staticmethod
    def register_subcommand(parser: _SubParsersAction) -> None:
        """Registers subcommands for device-related operations.

        Args:
            parser (_SubParsersAction): The parser object to add subcommands to.
        """
        device_parser = parser.add_parser("device", help="Device setting from owlite")
        device_parser.add_argument(
            "mode",
            choices=["ls", "add", "rm", "connect", "disconnect"],
            help="Device setting command",
        )

        device_parser.add_argument(
            "--name",
            "-n",
            type=str,
            default="NEST",
            help="device manager name",
        )
        device_parser.add_argument(
            "--url",
            "-u",
            type=str,
            help="device manager url",
        )

        device_parser.set_defaults(func=lambda args: DeviceCommand(args))


class DeviceCommand:
    """Handles device-specific commands in OwLite CLI."""

    def __init__(self, args: Namespace) -> None:
        """Initializes the DeviceCommand.

        Args:
            args: Arguments passed to the command.
        """
        self.args = args

    def run(self) -> None:
        """Executes the specified device-related operation based on the mode specified"""

        if self.args.mode == "ls":
            print_manager_list()
        elif self.args.mode == "add":
            add_manager(self.args.name, self.args.url.rstrip("/"))
        elif self.args.mode == "rm":
            remove_manager(self.args.name)
        elif self.args.mode == "connect":
            connect_device(self.args.name)
        elif self.args.mode == "disconnect":
            disconnect_device()
