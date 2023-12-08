"""Logger for OwLite"""
# pylint: disable=protected-access
import logging
import os

from onnx_graphsurgeon.logger import G_LOGGER

from owlite_core.constants import OWLITE_PREFIX


# pylint: disable=missing-function-docstring, too-few-public-methods
class Logger(logging.Logger):
    """The Logger class whose level can be only set via the environmental variable OWLITE_LOG_LEVEL"""

    ENV_VAR = "OWLITE_LOG_LEVEL"

    def ignore_warnings(self):
        """Return context manager to ignore warning.

        with log.ignore_warnings():
            log.warning("this warning would be ignored")

        Returns:
            _WarningFilterContext instatnce
        """

        class _WarningFilterContext:
            def __init__(self, logger) -> None:
                self.logger = logger
                self.warning_filter = None

            def __enter__(self):
                class WarningFilter(logging.Filter):
                    """Class to filter warnings"""

                    def filter(self, record):
                        return record.levelno < DEBUG_WARNING

                self.warning_filter = WarningFilter()
                self.logger.addFilter(self.warning_filter)
                return self.logger

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.logger.removeFilter(self.warning_filter)

        return _WarningFilterContext(self)

    def debug_warning(self, msg, *args, **kwargs):
        if self.isEnabledFor(DEBUG_WARNING):
            self._log(DEBUG_WARNING, msg, args, **kwargs)

    # pylint: disable=access-member-before-definition, attribute-defined-outside-init
    @property
    def level(self) -> int:
        if hasattr(self, "_level"):
            return self._level
        level_from_env = os.getenv(Logger.ENV_VAR, None)
        if level_from_env is None:
            self._level = logging.INFO
        elif all(c.isdigit() for c in level_from_env):
            self._level = int(level_from_env)
        else:
            self._level = logging._nameToLevel.get(level_from_env, logging.INFO)
        G_LOGGER.severity = self._level
        return self._level

    # pylint: disable=unused-argument
    @level.setter
    def level(self, value):
        return


if "owlite" not in logging.getLogger().manager.loggerDict:
    DEBUG_WARNING = 15
    ULTRA_VERBOSE = -10
    logging.addLevelName(DEBUG_WARNING, "DEBUG] [WARNING")
    logging.addLevelName(ULTRA_VERBOSE, "ULTRA_VERBOSE")

    log = Logger("owlite")

    formatter = (
        logging.Formatter(
            f"{OWLITE_PREFIX}%(pathname)s:%(lineno)d [%(levelname)s] %(message)s"
        )
        if log.level <= ULTRA_VERBOSE
        else logging.Formatter(f"{OWLITE_PREFIX}[%(levelname)s] %(message)s")
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    log.addHandler(stream_handler)


def suppress_owlite_warnings(cls):
    """
    A decorator to suppress owlite warnings during the initialization of class.

    Parameters:
    - cls: The class to which the decorator is applied.

    Returns:
    - The decorated class.
    """
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        with log.ignore_warnings():
            original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    return cls
