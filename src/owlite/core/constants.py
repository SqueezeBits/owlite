import os
from pathlib import Path
from typing import Literal

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
OWLITE_SETTINGS_FORMAT_VERSION = Version("2.1")
OWLITE_VERSION = Version("2.5.0")

# pylint: disable-next=invalid-name
SUPPORTED_QUALCOMM_DEVICES = Literal[
    "SA7255P ADP",
    "SA8255 (Proxy)",
    "SA8295P ADP",
    "SA8650 (Proxy)",
    "SA8775 (Proxy)",
    "SA8775P ADP",
    "Snapdragon 8cx Gen 3 CRD",
    "Snapdragon X Elite CRD",
    "Snapdragon X Plus 8-Core CRD",
    "QCS6490 (Proxy)",
    "QCS8275 (Proxy)",
    "QCS8550 (Proxy)",
    "QCS9075 (Proxy)",
    "RB3 Gen 2 (Proxy)",
    "Samsung Galaxy A73 5G",
    "Samsung Galaxy S21",
    "Samsung Galaxy S21 (Family)",
    "Samsung Galaxy S21 Ultra",
    "Samsung Galaxy S21+",
    "Samsung Galaxy S22 (Family)",
    "Samsung Galaxy S22 5G",
    "Samsung Galaxy S22 Ultra 5G",
    "Samsung Galaxy S22+ 5G",
    "Samsung Galaxy S23",
    "Samsung Galaxy S23 (Family)",
    "Samsung Galaxy S23 Ultra",
    "Samsung Galaxy S23+",
    "Samsung Galaxy S24",
    "Samsung Galaxy S24 (Family)",
    "Samsung Galaxy S24 Ultra",
    "Samsung Galaxy S24+",
    "Samsung Galaxy S25",
    "Samsung Galaxy S25 (Family)",
    "Samsung Galaxy S25 Ultra",
    "Samsung Galaxy S25+",
    "Samsung Galaxy Tab S8",
    "Snapdragon 8 Elite QRD",
    "Xiaomi 12",
    "Xiaomi 12 (Family)",
    "Xiaomi 12 Pro",
    "QCS8450 (Proxy)",
    "XR2 Gen 2 (Proxy)",
]
