import os
from pathlib import Path
from typing import Optional

# default cache
owlite_cache_dir = os.path.expanduser(
    os.getenv(
        "OWLITE_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "owlite"),
    )
)


def read_text(path: Path) -> Optional[str]:
    """
    Reads text from a file specified by the given path.

    Args:
        path (Path): The path to the file.

    Returns:
        Optional[str]: Text read from the file or None if the file is not found.
    """
    try:
        text = path.read_text(encoding="utf-8")
        text = text.replace("\r", "").replace("\n", "").strip()
        return text
    except FileNotFoundError:
        return None
