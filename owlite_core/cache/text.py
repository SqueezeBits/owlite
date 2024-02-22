from pathlib import Path
from typing import Optional

from packaging.version import Version

from ..constants import OWLITE_SETTINGS_FORMAT_VERSION
from ..logger import log


def write_text(path: Path, text: str) -> None:
    """Writes text to a file specified by the given path.

    Args:
        path (Path): The path to the file.
        text (str): The text to write.
    """
    versioned_text = f"{text}@v{OWLITE_SETTINGS_FORMAT_VERSION}"
    path.write_text(versioned_text, encoding="utf-8")


def read_text(path: Path) -> Optional[str]:
    """Reads text from a file specified by the given path.

    Args:
        path (Path): The path to the file.

    Returns:
        Optional[str]: Text read from the file or None if the file is not found.
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
    cache_version = Version(version=version)
    current_version = Version(OWLITE_SETTINGS_FORMAT_VERSION)
    if cache_version.major < current_version.major:
        log.error("version not matched. remove ~/.cache/owlite file and retry")  # UX
        raise RuntimeError("version not matched")
    return text
