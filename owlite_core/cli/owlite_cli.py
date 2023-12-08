"""CLI script to manage OwLite commands using argparse."""
from argparse import ArgumentParser

from .device_commands import DeviceCommands
from .url_commands import UrlCommands
from .user_commands import UserCommands


def main() -> None:
    """Main function to set up and run OwLite CLI commands."""

    parser = ArgumentParser("owlite", usage="owlite <command> [<args>]")
    commands_parser = parser.add_subparsers(help="owlite command helpers")

    # Register commands
    UserCommands.register_subcommand(commands_parser)
    DeviceCommands.register_subcommand(commands_parser)
    UrlCommands.register_subcommand(commands_parser)

    # pylint: disable-next=too-few-public-methods, missing-class-docstring
    class _Default:
        # pylint: disable-next=missing-function-docstring
        def run(self) -> None:
            parser.print_help()

    parser.set_defaults(func=lambda _: _Default())
    args = parser.parse_args()

    owlite_cli = args.func(args)
    try:
        owlite_cli.run()
    except Exception:  # pylint: disable=broad-exception-caught
        return


if __name__ == "__main__":
    main()
