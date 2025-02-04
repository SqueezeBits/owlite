import os
from pathlib import Path

from packaging.version import Version

OWLITE_HOME_PATH = Path(os.getenv("OWLITE_HOME", os.path.join(os.getcwd(), "owlite"))).resolve()

OWLITE_REPORT_URL = "https://squeezebits.zendesk.com/hc/en-us"

OWLITE_FRONT_BASE_URL = "https://owlite.ai"
OWLITE_MAIN_API_BASE_URL = "https://api.owlite.ai"
OWLITE_DOVE_API_BASE_URL = "https://dove.owlite.ai"

NEST_URL = "https://nest.owlite.ai"

OWLITE_GIT_REPO_URL = "https://github.com/SqueezeBits/owlite"

OWLITE_API_DEFAULT_TIMEOUT = int(os.environ.get("OWLITE_API_DEFAULT_TIMEOUT", "300"))

OWLITE_CALIBRATOR_HISTOGRAM_SIZE = int(os.environ.get("OWLITE_CALIBRATOR_HISTOGRAM_SIZE", "2048"))
OWLITE_CALIBRATION_ENABLE_GRAD = bool(os.environ.get("OWLITE_CALIBRATION_ENABLE_GRAD", False))

FX_CONFIGURATION_FORMAT_VERSION = Version("1.3")
OWLITE_SETTINGS_FORMAT_VERSION = Version("2.0")
OWLITE_VERSION = Version("2.3.0")
