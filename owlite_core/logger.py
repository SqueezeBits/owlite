import logging

from .constants import OWLITE_PREFIX

log = logging.getLogger("owlite")

formatter = logging.Formatter(f"{OWLITE_PREFIX}[%(levelname)s] %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

log.addHandler(stream_handler)
log.setLevel(logging.INFO)
