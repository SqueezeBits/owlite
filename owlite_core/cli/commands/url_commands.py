"""Handles URL-related commands in OwLite CLI using argparse"""
# pylint: disable=unnecessary-lambda, too-few-public-methods
from argparse import Namespace, _SubParsersAction

from .. import BaseOwLiteCLICommand
from ..url import delete_base_url, print_base_urls, save_base_url


class UrlCommands(BaseOwLiteCLICommand):
    """Handles URL-related commands in OwLite CLI."""

    @staticmethod
    def register_subcommand(parser: _SubParsersAction) -> None:
        """Registers subcommands for URL-related operations.

        Args:
            parser (_SubParsersAction): The parser object to add subcommands to.
        """
        url_parser = parser.add_parser("url", help="Set OwLite API base")
        url_parser.add_argument(
            "mode",
            choices=["add", "rm", "ls"],
            help="Device setting command",
        )

        url_parser.add_argument(
            "--name",
            "-n",
            type=str,
            help="API base name",
        )
        url_parser.add_argument(
            "--url",
            "-u",
            type=str,
            help="API base url",
        )

        url_parser.set_defaults(func=lambda args: UrlCommand(args))


class UrlCommand:
    """Handles URL-related commands in OwLite CLI"""

    def __init__(self, args: Namespace) -> None:
        """Initializes the UrlCommand.

        Args:
            args: Arguments passed to the command.
        """
        self.args = args

    def run(self) -> None:
        """Executes the specified URL-related operation"""

        if self.args.mode == "add":
            save_base_url(self.args.name, self.args.url.rstrip("/"))
        elif self.args.mode == "rm":
            delete_base_url(self.args.name)
        elif self.args.mode == "ls":
            print_base_urls()
