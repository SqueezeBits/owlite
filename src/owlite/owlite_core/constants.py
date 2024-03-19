import os

OWLITE_HOME = os.path.join(os.getenv("OWLITE_HOME", os.path.join(os.getcwd(), "owlite")))

OWLITE_REPORT_URL = "https://tally.so/r/mOl5Zk"

OWLITE_FRONT_BASE_URL = "https://owlite.ai"
OWLITE_MAIN_API_BASE_URL = "https://api.owlite.ai"
OWLITE_DOVE_API_BASE_URL = "https://dove.owlite.ai"

NEST_URL = "https://nest.owlite.ai"

OWLITE_GIT_REPO_URL = "https://github.com/SqueezeBits/owlite"

OWLITE_API_DEFAULT_TIMEOUT = int(os.environ.get("OWLITE_API_DEFAULT_TIMEOUT", "15"))

FX_CONFIGURATION_FORMAT_VERSION = "1.1"
OWLITE_SETTINGS_FORMAT_VERSION = "1.1"
OWLITE_VERSION = "1.2.2"
