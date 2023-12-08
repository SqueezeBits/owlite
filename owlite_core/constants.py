"""Constants for file, paths."""
import os

OWLITE_REPO_PATH = os.path.join(os.getenv("OWLITE_REPO_DIR", os.path.join(os.getcwd(), "owlite")))

OWLITE_PREFIX = "\033[1m\u001b[38;5;208mOwLite\u001b[0m\033[0m "

OWLITE_REPORT_URL = "https://tally.so/r/mOl5Zk"

OWLITE_FRONT_BASE_URL = "https://owlite.ai"
OWLITE_MAIN_API_BASE_URL = "https://api.owlite.ai"
OWLITE_DOVE_API_BASE_URL = "https://dove.owlite.ai"

OWLITE_DEFAULT_DEVICE_MANAGER = "https://nest.owlite.ai"

OWLITE_API_DEFAULT_TIMEOUT = int(os.environ.get("OWLITE_API_DEFAULT_TIMEOUT", "15"))
