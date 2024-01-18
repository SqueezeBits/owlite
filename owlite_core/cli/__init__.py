"""BaseCLICommand class for using owlite"""
from abc import ABC, abstractmethod
from argparse import _SubParsersAction


class BaseOwLiteCLICommand(ABC):
    """Abstract base class defining the structure for OwLite CLI commands"""

    @staticmethod
    @abstractmethod
    def register_subcommand(parser: _SubParsersAction) -> None:
        """Abstract method to register subcommands.

        This method should be implemented by subclasses to register subcommands with the provided parser.

        Args:
            parser (_SubParsersAction): The ArgumentParser for registering subcommands.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def run(self) -> None:
        """Abstract method to execute the command logic.

        This method should be implemented by subclasses to define the logic executed when the command runs.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError()
