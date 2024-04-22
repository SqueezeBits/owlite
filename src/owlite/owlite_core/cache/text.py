from pathlib import Path

from packaging.version import Version

from ..constants import OWLITE_SETTINGS_FORMAT_VERSION
from ..logger import log
from . import OWLITE_CACHE_PATH


def write_text(path: Path, text: str) -> None:
    """Write text to a file specified by the given path.

    Args:
        path (Path): The path to the file.
        text (str): The text to write.
    """
    versioned_text = f"{text}@v{OWLITE_SETTINGS_FORMAT_VERSION}"
    path.write_text(versioned_text, encoding="utf-8")


def read_text(path: Path) -> str | None:
    """Read text from a file specified by the given path.

    Args:
        path (Path): The path to the file.

    Returns:
        str | None: Text read from the file or None if the file is not found.
    """
    try:
        cached_text = path.read_text(encoding="utf-8")
        cached_text = cached_text.replace("\r", "").replace("\n", "").strip()
    except FileNotFoundError:
        return None

    if "@v" not in cached_text:
        text = cached_text
        version = "1.0"
        path.write_text(f"{cached_text}@v{version}", encoding="utf-8")
    else:
        text, version = cached_text.rsplit("@v", 1)
    if Version(version).major < OWLITE_SETTINGS_FORMAT_VERSION.major:
        log.error(
            f"The cache version ({Version(version)}) is not supported. "
            f"Please remove the cache file in {OWLITE_CACHE_PATH} and retry"
        )  # UX
        raise RuntimeError("Version is not supported")
    return text
