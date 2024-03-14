# pylint: disable=protected-access
import logging
import os
from typing import Any, Callable, Optional, TypeVar, cast

from onnx_graphsurgeon.logger import G_LOGGER

DEBUG_WARNING = 15
ULTRA_VERBOSE = -10


# pylint: disable=missing-function-docstring, too-few-public-methods
class Logger(logging.Logger):
    """The Logger class whose level can be only set via the environmental variable OWLITE_LOG_LEVEL"""

    ENV_VAR = "OWLITE_LOG_LEVEL"

    def ignore_warnings(self) -> Any:
        """Return context manager to ignore warning.

        with log.ignore_warnings():
            log.warning("this warning would be ignored")

        Returns:
            _WarningFilterContext instatnce
        """

        class _WarningFilterContext:
            def __init__(self, logger: logging.Logger) -> None:
                self.logger = logger
                self.warning_filter: Optional[logging.Filter] = None

            def __enter__(self) -> logging.Logger:
                class WarningFilter(logging.Filter):
                    """Class to filter warnings"""

                    def filter(self, record: logging.LogRecord) -> bool:
                        return record.levelno < DEBUG_WARNING

                self.warning_filter = WarningFilter()
                self.logger.addFilter(self.warning_filter)
                return self.logger

            def __exit__(self, exc_type: type[BaseException], exc_val: BaseException, exc_tb: Any) -> None:
                if self.warning_filter:
                    self.logger.removeFilter(self.warning_filter)

        return _WarningFilterContext(self)

    def debug_warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self.isEnabledFor(DEBUG_WARNING):
            self._log(DEBUG_WARNING, msg, args, **kwargs)

    # pylint: disable=access-member-before-definition, attribute-defined-outside-init
    @property
    def level(self) -> int:
        if hasattr(self, "_level"):
            return self._level
        level_from_env = os.getenv(Logger.ENV_VAR, None)
        if level_from_env is None:
            self._level = logging.INFO  # type: int
        elif all(c.isdigit() for c in level_from_env):
            self._level = int(level_from_env)
        else:
            self._level = logging._nameToLevel.get(level_from_env, logging.INFO)
        G_LOGGER.severity = self._level
        return self._level

    # pylint: disable=unused-argument
    @level.setter
    def level(self, value: Any) -> None:
        return


class OwLiteFormatter(logging.Formatter):
    """Custom log formatter for OwLite application.

    This formatter customizes log messages by adding color-coded level names and an OwLite prefix.
    It uses ANSI escape codes for color representation.

    Args:
        format_str (str): Log format string.

    Attributes:
        FORMATS (dict): A dictionary containing ANSI escape codes for different log levels.
        reset (str): ANSI escape code to reset colors to default.
        owlite_prefix (str): Prefix for OwLite log messages.

    """

    def __init__(self, format_str: str) -> None:
        super().__init__(format_str)

    def format(self, record: logging.LogRecord) -> str:
        log_format = self.FORMATS.get(record.levelno, "")
        colored_levelname = f"{log_format}[{record.levelname}]{self.reset}"

        record.levelname = colored_levelname

        return f"{self.owlite_prefix}{super().format(record)}"

    FORMATS = {
        logging.WARNING: "\x1b[38;2;255;212;0m",  # Yellow color for WARNING
        logging.ERROR: "\x1b[38;2;255;40;40m",  # Red color for ERROR
        logging.DEBUG: "\x1b[38;2;123;131;191m",  # Some shade of blue color for DEBUG
        DEBUG_WARNING: "\x1b[38;2;175;0;2151m",  # DarkViolet color for DEBUG WARNING
    }

    reset = "\x1b[0m"
    owlite_prefix = f"\x1b[38;2;238;120;31;1mOwLite {reset}"


if "owlite" not in logging.getLogger().manager.loggerDict:
    logging.addLevelName(DEBUG_WARNING, "DEBUG WARNING")
    logging.addLevelName(ULTRA_VERBOSE, "ULTRA_VERBOSE")

    log = Logger("owlite")

    formatter = (
        OwLiteFormatter("%(pathname)s:%(lineno)d %(levelname)s %(message)s")
        if log.level <= ULTRA_VERBOSE
        else OwLiteFormatter("%(levelname)s %(message)s")
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    log.addHandler(stream_handler)

else:
    log = cast(Logger, logging.getLogger().manager.loggerDict["owlite"])


T = TypeVar("T")


def suppress_owlite_warnings(cls: T) -> T:
    """
    A decorator to suppress owlite warnings during the initialization of class.

    Parameters:
    - cls: The class to which the decorator is applied.

    Returns:
    - The decorated class.
    """
    original_init: Callable[..., None] = cls.__init__  # type: ignore[misc]

    def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
        assert isinstance(log, Logger)
        with log.ignore_warnings():
            original_init(self, *args, **kwargs)

    cls.__init__ = new_init  # type: ignore[misc]
    return cls
