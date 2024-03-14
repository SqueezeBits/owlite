import os

import requests
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper

from ..owlite_core.logger import log


def upload_file_to_url(file_path: str, dst_url: str) -> None:
    """
    Upload file to destination URL via http request.

    Args:
        file_path (str): path to file
        dst_url (str): url to upload

    Raises:
        FileNotFoundError: when file does not exists at given path

        HTTPError: when request was not successful
    """
    log.info(f"Uploading {file_path}")  # UX

    if not os.path.exists(file_path):
        log.error(f"Cannot upload {file_path} as it is not found")  # UX
        raise FileNotFoundError("File not found")

    total = os.path.getsize(file_path)
    with open(file_path, "rb") as file:
        with tqdm(
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            reader_wrapper = CallbackIOWrapper(progress_bar.update, file, "read")
            # pylint: disable-next=missing-timeout
            resp = requests.put(dst_url, data=reader_wrapper)
            if not resp.ok:
                resp.raise_for_status()

        log.info("Uploading done")  # UX


def download_file_from_url(file_url: str, path_to_save: str) -> None:
    """
    Download file from URL via http request, note that this function will overwrite a file with
    downloaded file content if a file already exists at given path.

    Args:
        file_url: URL of a file to download

        path_to_save: path to save downloaded file

    Raises:
        HTTPError: when request was not successful
    """
    log.info(f"Downloading file at {path_to_save}")  # UX

    if os.path.exists(path_to_save):
        log.warning(f"The existing file at {path_to_save} will be overwritten")  # UX

    resp = requests.get(file_url, stream=True)  # pylint: disable=missing-timeout
    total = int(resp.headers.get("content-length", 0))
    with open(path_to_save, "wb") as file, tqdm(
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

        if not resp.ok:
            resp.raise_for_status()

    log.info("Downloading done")  # UX
