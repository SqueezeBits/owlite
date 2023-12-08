# mypy: disable-error-code = attr-defined
"""API wrapper module for devices"""
import os
import signal
import sys
import time
from enum import Enum
from typing import Any

from owlite_core.cli.device import OWLITE_DEVICE_NAME
from owlite_core.constants import (
    OWLITE_FRONT_BASE_URL,
    OWLITE_REPORT_URL,
)

from ...logger import log
from ...utils.file_utils import download_file_from_url, upload_file_to_url
from ..api_base import DEVICE_API_BASE


class BenchmarkStatus(Enum):
    """TensorRT benchmark job status"""

    IDLE = 0
    PRE_FETCHING = 1
    BENCHMARKING = 2
    BENCHMARK_DONE = 3


def request_trt_benchmark(benchmark_key: str, bin_path: str) -> None:
    """Uploads ONNX weight binary file and request TensorRT benchmark.

    Args:
        benchmark_key (str): A key to identify benchmark job.
        bin_path (str): The path of a ONNX weight binary file.

    Raises:
        FileNotFoundError: When bin file does not exists at given path.
        ValueError: When device is not set.
        HTTPError: When request was not successful.
    """

    if not os.path.exists(bin_path):
        log.error(
            f"Unable to locate the ONNX bin file at the specified path: {bin_path}. "
            "Please ensure the file exists and the path is accurate. "
            "If the file is missing, recreate the ONNX file and retry"
        )
        raise FileNotFoundError("ONNX bin file not found")

    device_name = OWLITE_DEVICE_NAME
    if device_name is None:
        log.error("Connected device not found. Please connect device by 'owlite device connect'")
        raise ValueError("Device not found")

    payload = {
        "device_name": device_name,
        "benchmark_key": benchmark_key,
    }

    resp = DEVICE_API_BASE.post("/devices/jobs/export", json=payload)
    assert isinstance(resp, dict)

    file_dest_url = resp["bin_file_url"]

    file_upload_resp = upload_file_to_url(bin_path, file_dest_url)
    if not file_upload_resp.ok:
        file_upload_resp.raise_for_status()


def get_benchmark_queue_info(benchmark_key: str) -> dict:
    """Gets information of an experiment.

    Args:
        benchmark_key (str): A key to identify benchmark job.

    Returns:
        dict: Queueing information of an experiment.

    Raises:
        HTTPError: When request was not successful.
    """

    device_name = OWLITE_DEVICE_NAME
    if device_name is None:
        log.error("Connected device not found. Please connect device by 'owlite device connect'")
        raise RuntimeError("Device not found")

    payload = {
        "device_name": device_name,
        "benchmark_key": benchmark_key,
    }

    res = DEVICE_API_BASE.post("/devices/jobs/queue", json=payload)
    assert isinstance(res, dict)

    return res


def poll_run_benchmark(project_id: str, benchmark_key: str) -> None:
    """Polls for TensorRT benchmark result.

    Args:
        project_id (str): The id of a project.
        benchmark_key (str): A key to identify benchmark job.

    Raises:
        ValueError: When unexpected signal is caught by SIGINT handler.
        RuntimeError: When error occurred during TensorRT execution.
    """

    def sigint_handler(sig: signal.Signals, frame: Any) -> None:
        if sig != signal.SIGINT:
            raise ValueError(f"Unexpected signals: {sig} (frame={frame})")
        print("")
        log.info(
            f"Exit from current experiment. "
            f"Continue creating config at "
            f"{OWLITE_FRONT_BASE_URL}/project/detail/{project_id}"
        )
        sys.exit(sig)

    original_sigint_handler = signal.signal(signal.SIGINT, sigint_handler)  # type: ignore

    log.info("Polling for benchmark result, you are free to CTRL-C away")

    count = 0
    info = get_benchmark_queue_info(benchmark_key)
    benchmark_status = info["benchmark_status"]
    in_progress = (
        BenchmarkStatus.PRE_FETCHING.value,
        BenchmarkStatus.BENCHMARKING.value,
    )
    while True:
        if count % 5 == 0:
            info = get_benchmark_queue_info(benchmark_key)
            new_status = info["benchmark_status"]

            if new_status < 0:
                print("")
                log.error(
                    "Runtime error occurred during TensorRT engine execution or benchmark. Please try again. "
                    f"If the problem persists, please report us at {OWLITE_REPORT_URL} for further assistance"
                )
                raise RuntimeError("Benchmarking error")

            if benchmark_status != new_status and new_status in in_progress:
                benchmark_status = new_status
                count = 0

            elif new_status == BenchmarkStatus.BENCHMARK_DONE.value:
                print("\nBenchmarking done")
                signal.signal(signal.SIGINT, original_sigint_handler)
                return

        if benchmark_status in in_progress:
            if benchmark_status == BenchmarkStatus.PRE_FETCHING.value and info["prefetch"] is not None:
                message = f"Your position in the queue: {info['prefetch']} {'. ' * (count % 4)}"

            else:
                dots_before = "." * count
                owl_emoji = "\U0001F989"
                dots_after = "." * (19 - count)

                message = f"[{dots_before}{owl_emoji}{dots_after}]"

            print(f"\r{message:<50}", end="", flush=True)

        count = (count + 1) % 20
        time.sleep(2)


def download_trt_engine(benchmark_key: str, path_to_save: str) -> None:
    """Downloads built TensorRT engine.

    Args:
        benchmark_key (str): A key to identify benchmark job.
        path_to_save (str): The path to save downloaded TensorRT engine.

    Raises:
        RuntimeError: When device is not set.
        HTTPError: When request was not successful.
    """
    device_name = OWLITE_DEVICE_NAME
    if device_name is None:
        log.error("Device is not set. Please set device and try again")
        raise RuntimeError("Device not found")

    payload = {
        "device_name": device_name,
        "benchmark_key": benchmark_key,
    }
    resp = DEVICE_API_BASE.post("/devices/trt", json=payload)
    assert isinstance(resp, dict)

    file_url = resp["trt_engine_url"]

    download_file_from_url(file_url, path_to_save)
