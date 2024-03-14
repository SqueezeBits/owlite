# pylint: disable=duplicate-code, too-many-public-methods
import json
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass, fields
from functools import cached_property
from typing import Any, Optional, Union

import onnx
import requests
from torch.fx.graph_module import GraphModule

from ..backend.onnx.export import export
from ..backend.signature import DynamicSignature, Signature
from ..options import DynamicAxisOptions, ONNXExportOptions
from ..owlite_core.api_base import MAIN_API_BASE, APIBase
from ..owlite_core.api_enums import BenchmarkStatus, PricePlan
from ..owlite_core.cli.api.login import whoami
from ..owlite_core.cli.device import connect_free_device
from ..owlite_core.constants import OWLITE_API_DEFAULT_TIMEOUT, OWLITE_REPORT_URL
from ..owlite_core.logger import log
from ..owlite_core.owlite_settings import OWLITE_SETTINGS
from .utils import download_file_from_url, upload_file_to_url

DEVICE_API_BASE: APIBase = APIBase(
    OWLITE_SETTINGS.connected_device.manager.url
    if OWLITE_SETTINGS.connected_device
    else OWLITE_SETTINGS.base_url.NEST,  # type: ignore
    "OWLITE_DEVICE_API_BASE",
)


@dataclass
class BenchmarkResult:
    """TensorRT benchmark result"""

    name: str
    device_name: str
    latency: float
    vram: float


@dataclass
class Benchmarkable:
    """Base protocol for objects that can request TensorRT benchmark"""

    name: str

    @cached_property
    def plan(self) -> PricePlan:
        """Pricing plan of current user"""
        userinfo = whoami()
        return PricePlan(userinfo.plan)

    @property
    def device(self) -> str:
        """Gets the connected device's name.

        If none found, connects to a free device for free plan users; raises an error for paid plan users.

        Returns:
            str: The name of the connected device.

        Raises:
            RuntimeError: If no device is found and the user is on a paid plan.
        """
        connected_device = OWLITE_SETTINGS.connected_device
        if OWLITE_SETTINGS.connected_device is None:
            if not self.plan.paid:
                connected_device = connect_free_device()
            else:
                log.error(
                    "Device not found. Please connect to a device using 'owlite device connect --name (name)'"
                )  # UX
                raise RuntimeError("Device not found")
        assert connected_device
        return connected_device.name

    @property
    def input_signature(self) -> Optional[Union[Signature, DynamicSignature]]:
        """Input signature of model"""
        raise NotImplementedError()

    @property
    def url(self) -> str:
        """The URL to the relevant page for this object"""
        raise NotImplementedError()

    @property
    def home(self) -> str:
        """The directory path for writing outputs produced by this object"""
        raise NotImplementedError()

    @property
    def label(self) -> str:
        """A unique label for this object"""
        raise NotImplementedError()

    @property
    def onnx_path(self) -> str:
        """The file path for writing ONNX proto"""
        return os.path.join(self.home, f"{self.label}.onnx")

    @property
    def bin_path(self) -> str:
        """The file path for writing ONNX weight"""
        return os.path.join(self.home, f"{self.label}.bin")

    @property
    def engine_path(self) -> str:
        """The file path for writing TensorRT engine"""
        return os.path.join(self.home, f"{self.label}.engine")

    @cached_property
    def benchmark_key(self) -> str:
        """The key for requesting benchmark"""
        if self.input_signature is None:
            log.error(
                "TensorRT benchmark requires the ONNX proto exported from your model. "
                "Call `owl.export` before calling `owl.benchmark`"
            )  # UX
            raise RuntimeError("Input signature not found")
        resp = MAIN_API_BASE.post(
            "/projects/runs/keys", json=self.payload(run_name=self.name, input_shape=json.dumps(self.input_signature))
        )
        assert isinstance(resp, str)
        return resp

    def export(
        self,
        model: GraphModule,
        args: Optional[tuple[Any, ...]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        dynamic_axis_options: Optional[DynamicAxisOptions] = None,
        onnx_export_options: Optional[ONNXExportOptions] = None,
    ) -> onnx.ModelProto:
        """Exports the graph module `model` into ONNX.

        Args:
            model (GraphModule): a graph module
            args (Optional[tuple[Any, ...]], optional): the arguments to be passed to the model. Defaults to None.
            kwargs (Optional[dict[str, Any]], optional): the keyword arguments to be passed to the model.
                Defaults to None.
            dynamic_axis_options (Optional[DynamicAxisOptions], optional): Optional dynamic export options.
                Defaults to None.
            onnx_export_options (Optional[ONNXExportOptions], optional): Optional ONNX export options.
                Defaults to None.

        Returns:
            onnx.ModelProto: ONNX proto of model
        """
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        export(
            model,
            (*args, kwargs),
            self.onnx_path,
            dynamic_axis_options=dynamic_axis_options,
            **asdict(onnx_export_options or ONNXExportOptions()),
        )
        log.info(f"{type(self).__name__} ONNX saved at {self.onnx_path}")  # UX
        return onnx.load(self.onnx_path, load_external_data=False)

    def upload(
        self,
        proto: onnx.ModelProto,
        model: Optional[GraphModule],
    ) -> None:
        """Uploads the model.

        Args:
            proto (onnx.ModelProto): ONNX proto of a model
            model (GraphModule): A model converted into a graph module
        """
        raise NotImplementedError()

    def orchestrate_trt_benchmark(self) -> None:
        """Orchestrates the end-to-end TensorRT benchmark pipeline."""
        if self.device is None:
            log.warning(
                "Cannot initiate TensorRT benchmark. Please connect to a device first "
                "using 'owlite device connect --name (name)'"
            )  # UX
            return

        log.info(f"Benchmark initiated for the {self}")  # UX
        self.request_trt_benchmark()
        log.info("TensorRT benchmark requested")  # UX
        self.poll_trt_benchmark()

        result = self.get_trt_benchmark_result()
        indent = " " * 14
        log.info(
            f"{type(self).__name__}: {result.name}\n"
            f"{indent}Latency: {result.latency} (ms) on {result.device_name}\n"
            f"{indent}For more details, visit {self.url}"
        )  # UX
        if self.plan.paid:
            self.download_trt_engine()
        else:
            log.info(
                "Your current account plan (free) is not eligible for downloading TensorRT engines. "
                "Please consider upgrading your plan for the seamless experience that OwLite can provide. "
                f"However, you can still convert the ONNX at {self.onnx_path} into a TensorRT engine by yourself"
            )  # UX
        self.clear_trt_engine()

    def request_trt_benchmark(self) -> None:
        """Requests TensorRT benchmark.

        Raises:
            ValueError: When device is not set.
            HTTPError: When request was not successful.
        """

        resp = DEVICE_API_BASE.post(
            "/devices/jobs/assign",
            json={
                "device_name": self.device,
                "benchmark_key": self.benchmark_key,
            },
        )
        assert isinstance(resp, str)
        log.debug(f"request_trt_benchmark received {resp}")

    def get_benchmark_queue(self) -> dict:
        """Gets information of an experiment.
            If user's plan is upper than free plan, uploads model weights to device manager.

        Returns:
            dict: Queueing information of an experiment.

        Raises:
            HTTPError: When request was not successful.
        """
        res = DEVICE_API_BASE.post(
            "/devices/jobs/queue",
            json={
                "device_name": self.device,
                "benchmark_key": self.benchmark_key,
            },
        )
        assert isinstance(res, dict)
        log.debug(f"get_benchmark_queue received {res}")

        return res

    def upload_weight_file(self, bin_url: str) -> None:
        """Uploads ONNX weight binary file.

        Args:
            bin_url (str): Url to upload ONNX weight

        Raises:
            FileNotFoundError: When bin file does not exists at given path.
        """
        print()
        log.info("Uploading ONNX model weight to optimize the TensorRT engine")  # UX

        if not os.path.exists(self.bin_path):
            log.error(
                f"Missing ONNX weight file at {self.bin_path}. You may need to retry exporting your model to ONNX "
                "using `owl.export`"
            )  # UX
            raise FileNotFoundError("ONNX bin file not found")

        assert isinstance(bin_url, str)
        upload_file_to_url(self.bin_path, bin_url)

    def poll_trt_benchmark(self) -> None:
        """Polls for TensorRT benchmark result.

        Raises:
            ValueError: When unexpected signal is caught by SIGINT handler.
            RuntimeError: When error occurred during TensorRT execution.
        """

        def sigint_handler(sig: signal.Signals, frame: Any) -> None:
            if sig != signal.SIGINT:
                raise ValueError(f"Unexpected signals: {sig} (frame={frame})")
            log.info("\nMoving away from the polling. The benchmark will still run in the background")  # UX
            sys.exit(sig)

        original_sigint_handler = signal.signal(signal.SIGINT, sigint_handler)  # type: ignore

        log.info(
            "Polling for benchmark result. You are free to CTRL-C away. When it is done, you can find the results at "
            f"{self.url}"
        )  # UX

        count = 0
        benchmark_status = BenchmarkStatus.IDLE
        upload_weight = self.plan.paid
        while True:
            if count % 5 == 0:
                info = self.get_benchmark_queue()
                new_status = BenchmarkStatus(info.get("status", -999))
                if new_status == BenchmarkStatus.STATUS_NOT_FOUND:
                    print()
                    log.error(
                        "Benchmarking failed with an unexpected error. Please try again, and if the problem "
                        f"persists, please report the issue at {OWLITE_REPORT_URL} for further assistance"
                    )  # UX
                    raise RuntimeError(f"Benchmarking failed with status code {new_status}")

                bin_url = info.get("url", "")
                if upload_weight and new_status.PRE_FETCHING and len(bin_url) > 0:
                    self.upload_weight_file(bin_url)
                    upload_weight = False

                if benchmark_status != new_status and new_status.in_progress:
                    benchmark_status = new_status
                    count = 0

                elif new_status == BenchmarkStatus.BENCHMARK_DONE:
                    print("\nBenchmarking done")  # UX
                    signal.signal(signal.SIGINT, original_sigint_handler)
                    return

                elif new_status.failed:
                    _failed_msg = {
                        BenchmarkStatus.FETCHING_ERR: "Benchmarking failed with pre-fetching",
                        BenchmarkStatus.TIMEOUT_ERR: "Benchmarking failed with timeout",
                        BenchmarkStatus.BENCHMARK_ERR: "Benchmarking failed during benchmark",
                        BenchmarkStatus.WEIGHT_GEN_ERR: "Benchmarking failed with weight generation",
                    }
                    error_msg = _failed_msg.get(new_status, "Benchmarking failed with an unexpected error")
                    print()
                    log.error(
                        f"{error_msg}. Please try again, and if the problem "
                        f"persists, please report the issue at {OWLITE_REPORT_URL} for further assistance"
                    )  # UX
                    raise RuntimeError("Benchmarking failed")

            if benchmark_status.in_progress:
                job_position = info.get("pos", None)
                if benchmark_status == BenchmarkStatus.PRE_FETCHING and job_position is not None:
                    message = f"Your position in the queue: {job_position} {'. ' * (count % 4)}"  # UX

                else:
                    dots_before = "." * count
                    owl_emoji = "\U0001F989"
                    dots_after = "." * (19 - count)

                    message = f"[{dots_before}{owl_emoji}{dots_after}]"  # UX

                print(f"\r{message:<50}", end="", flush=True)

            count = (count + 1) % 20
            time.sleep(2)

    def get_trt_benchmark_result(self) -> BenchmarkResult:
        """Gets the benchmarking result.

        Returns:
            Optional[dict]: The information of an experiment if exists, None otherwise.

        Raises:
            HTTPError: When request was not successful.
        """

        try:
            res = MAIN_API_BASE.post("/projects/runs/info", json=self.payload(run_name=self.name))
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                log.error(
                    f"No such experiment: {self.name}. Please check if the experiment name is correct or create "
                    f"a new one at {self.url}"
                )  # UX
            raise e

        assert isinstance(res, dict)
        return BenchmarkResult(**{field.name: res[field.name] for field in fields(BenchmarkResult)})

    def download_trt_engine(self) -> None:
        """Downloads built TensorRT engine.

        Raises:
            RuntimeError: When device is not set.
            HTTPError: When request was not successful.
        """
        try:
            resp = DEVICE_API_BASE.post(
                "/devices/trt",
                json={
                    "device_name": self.device,
                    "benchmark_key": self.benchmark_key,
                },
            )
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                log.error(
                    "Missing TensorRT engine to download. "
                    "You may need to retry build the engine using `owl.benchmark`"
                    if self.plan.paid
                    else "The free plan doesn't support TensorRT engine download. "
                    "Upgrade to a higher plan to download the engine through OwLite with a seamless experience"
                )  # UX
            raise e
        assert isinstance(resp, dict)
        file_url = resp["trt_engine_url"]
        download_file_from_url(file_url, self.engine_path)

    def clear_trt_engine(self) -> None:
        """Clear created TensorRT engine on device."""
        log.debug(f"Clear TensorRT engine on device: {self.device}, benchmark_key: {self.benchmark_key}")
        resp = DEVICE_API_BASE.post(
            "/devices/clear",
            json={
                "device_name": self.device,
                "benchmark_key": self.benchmark_key,
            },
        )
        assert isinstance(resp, str)
        if len(resp) > 0:
            log.debug(f"Clear TensorRT engine with url: {resp}")
            del_res = requests.delete(resp, timeout=OWLITE_API_DEFAULT_TIMEOUT)
            if not del_res.ok:
                del_res.raise_for_status()

    def log(self, message: str) -> None:
        """Logs JSON-serialized metrics.
        Raises:
            HTTPError: When request was not successful.
        """
        resp = MAIN_API_BASE.post(
            "/projects/runs/update",
            json=self.payload(run_name=self.name, logs=message),
        )
        assert isinstance(resp, str)

    def payload(self, **kwargs: str) -> dict[str, str]:
        """The payload for API requests"""
        raise NotImplementedError()

    def __repr__(self) -> str:
        raise NotImplementedError()

    def __str__(self) -> str:
        return self.__repr__()
