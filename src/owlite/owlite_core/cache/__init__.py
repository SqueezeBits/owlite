import os

owlite_cache_dir = os.path.expanduser(
    os.getenv(
        "OWLITE_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "owlite"),
    )
)
