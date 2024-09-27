import os
from pathlib import Path

OWLITE_CACHE_PATH = Path(
    os.getenv(
        "OWLITE_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "owlite"),
    )
).resolve()

OWLITE_CACHE_PATH.mkdir(parents=True, exist_ok=True)
