"""CLI script to manage OwLite commands using argparse."""
from argparse import ArgumentParser

from ..constants import OWLITE_VERSION
from ..exceptions import DeviceError, LoginError
from ..logger import log
from .commands.device_commands import DeviceCommands
from .commands.url_commands import UrlCommands
from .commands.user_commands import UserCommands


def main() -> None:
    """Main function to set up and run OwLite CLI commands."""

    parser = ArgumentParser("owlite", usage="owlite <command> [<args>]")
    commands_parser = parser.add_subparsers(help="owlite command helpers")

    # Register commands
    UserCommands.register_subcommand(commands_parser)
    DeviceCommands.register_subcommand(commands_parser)
    UrlCommands.register_subcommand(commands_parser)

    parser.add_argument("--version", "-v", action="store_true", help="show OwLite CLI version")

    # pylint: disable-next=too-few-public-methods, missing-class-docstring
    class _Default:
        # pylint: disable-next=missing-function-docstring
        def run(self) -> None:
            parser.print_help()

    parser.set_defaults(func=lambda _: _Default())
    args = parser.parse_args()

    if args.version:
        log.info(f"OwLite version {OWLITE_VERSION}")  # UX
    else:
        owlite_cli = args.func(args)
        try:
            owlite_cli.run()
        except (LoginError, DeviceError) as e:
            log.debug(e)
            return


if __name__ == "__main__":
    main()
