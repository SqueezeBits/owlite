# pylint: disable=duplicate-code, too-many-public-methods, too-many-statements
import os
import signal
import sys
import time
from dataclasses import dataclass, fields
from functools import cached_property
from typing import Any

import onnx
import requests
from torch.fx.graph_module import GraphModule

from ..backend.onnx.export import export
from ..backend.signature import Signature
from ..enums.benchmark_status import BenchmarkStatus
from ..enums.price_plan import PricePlan
from ..options import DynamicAxisOptions, ONNXExportOptions
from ..owlite_core.api_base import MAIN_API_BASE, APIBase
from ..owlite_core.cache.device import Device
from ..owlite_core.cli.api.login import whoami
from ..owlite_core.constants import OWLITE_API_DEFAULT_TIMEOUT, OWLITE_REPORT_URL
from ..owlite_core.logger import log
from ..owlite_core.owlite_device_settings import OWLITE_DEVICE_SETTINGS
from ..owlite_core.owlite_settings import OWLITE_SETTINGS
from .utils import download_file_from_url, upload_file_to_url

DEVICE_API_BASE: APIBase = APIBase(
    OWLITE_DEVICE_SETTINGS.connected.manager.url if OWLITE_DEVICE_SETTINGS.connected else OWLITE_SETTINGS.base_url.NEST,
    "OWLITE_DEVICE_API_BASE",
)


@dataclass
class BenchmarkResult:
    """Benchmark result."""

    name: str
    device_name: str
    latency: float
    vram: float


@dataclass
class Benchmarkable:
    """Base protocol for objects that can request a benchmark."""

    name: str
    device: Device

    @cached_property
    def plan(self) -> PricePlan:
        """Pricing plan of current user."""
        userinfo = whoami()
        return PricePlan(userinfo.plan)

    @property
    def input_signature(self) -> Signature | None:
        """Input signature of model."""
        raise NotImplementedError()

    @property
    def url(self) -> str:
        """The URL to the relevant page for this object."""
        raise NotImplementedError()

    @property
    def home(self) -> str:
        """The directory path for writing outputs produced by this object."""
        raise NotImplementedError()

    @property
    def label(self) -> str:
        """A unique label for this object."""
        raise NotImplementedError()

    @property
    def onnx_path(self) -> str:
        """The file path for writing ONNX proto."""
        return os.path.join(self.home, f"{self.label}.onnx")

    @property
    def bin_path(self) -> str:
        """The file path for writing ONNX weight."""
        return os.path.join(self.home, f"{self.label}.bin")

    @property
    def engine_path(self) -> str:
        """The file path for writing the engine."""
        return os.path.join(self.home, f"{self.label}.engine")

    @cached_property
    def benchmark_key(self) -> str:
        """The key for requesting benchmark."""
        if self.input_signature is None:
            log.error(
                "Benchmark requires the ONNX proto exported from your model. "
                "Call `owl.export` before calling `owl.benchmark`"
            )  # UX
            raise RuntimeError("Input signature not found")
        resp = MAIN_API_BASE.post(
            "/projects/runs/keys", json=self.payload(run_name=self.name, input_shape=self.input_signature.dumps())
        )
        assert isinstance(resp, str)
        return resp

    def export(
        self,
        model: GraphModule,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
        dynamic_axis_options: DynamicAxisOptions | None = None,
        onnx_export_options: ONNXExportOptions | None = None,
    ) -> onnx.ModelProto:
        """Export the graph module `model` into ONNX.

        Args:
            model (GraphModule): a graph module
            args (tuple[Any, ...] | None, optional): the arguments to be passed to the model. Defaults to None.
            kwargs (dict[str, Any] | None, optional): the keyword arguments to be passed to the model.
                Defaults to None.
            dynamic_axis_options (DynamicAxisOptions | None, optional): Optional dynamic export options.
                Defaults to None.
            onnx_export_options (ONNXExportOptions | None, optional): Optional ONNX export options.
                Defaults to None.

        Returns:
            onnx.ModelProto: ONNX proto of model
        """
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        if dynamic_axis_options:
            assert self.input_signature
            self.input_signature.mark_dynamic_axes(dynamic_axis_options)
        if onnx_export_options is None:
            onnx_export_options = ONNXExportOptions.create(self.device)
        export(
            model,
            (*args, kwargs),
            self.onnx_path,
            dynamic_axis_options=dynamic_axis_options,
            **(onnx_export_options.model_dump()),
        )
        log.info(f"{type(self).__name__} ONNX saved at {self.onnx_path}")  # UX
        return onnx.load(self.onnx_path, load_external_data=False)

    def upload(
        self,
        proto: onnx.ModelProto,
        model: GraphModule,
    ) -> None:
        """Upload the model.

        Args:
            proto (onnx.ModelProto): ONNX proto of a model
            model (GraphModule): A model converted into a graph module
        """
        raise NotImplementedError()

    def orchestrate_benchmark(self) -> None:
        """Orchestrate the end-to-end benchmark pipeline."""
        if self.device is None:
            log.warning(
                "Cannot initiate benchmark. Please connect to a device first "
                "using 'owlite device connect --name (name)'"
            )  # UX
            return

        log.info(f"Benchmark initiated for the {self}")  # UX
        self.request_benchmark()
        log.info(f"Benchmark requested on '{self.device}'")  # UX
        self.poll_benchmark()

        result = self.get_benchmark_result()
        indent = " " * 14
        log.info(
            f"{type(self).__name__}: {result.name}\n"
            f"{indent}Latency: {result.latency} (ms) on {result.device_name}\n"
            f"{indent}For more details, visit {self.url}"
        )  # UX
        if self.plan.paid:
            self.download_engine()
        else:
            log.info(
                "Your current account plan (free) is not eligible for downloading engines. "
                "Please consider upgrading your plan for the seamless experience that OwLite can provide. "
                f"However, you can still convert the ONNX at {self.onnx_path} into a engine by yourself"
            )  # UX
        self.clear_engine()

    def request_benchmark(self) -> None:
        """Request benchmark.

        Raises:
            ValueError: When device is not set.
            HTTPError: When request was not successful.
        """
        resp = DEVICE_API_BASE.post(
            "/devices/jobs/assign",
            json={
                "device_name": self.device.name,
                "benchmark_key": self.benchmark_key,
            },
        )
        assert isinstance(resp, str)
        log.debug(f"request_benchmark received {resp}")

    def get_benchmark_queue(self) -> dict:
        """Get information of an experiment.

        If user's plan is upper than free plan, uploads model weights to device manager.

        Returns:
            dict: Queueing information of an experiment.

        Raises:
            HTTPError: When request was not successful.
        """
        res = DEVICE_API_BASE.post(
            "/devices/jobs/queue",
            json={
                "device_name": self.device.name,
                "benchmark_key": self.benchmark_key,
            },
        )
        assert isinstance(res, dict)
        log.debug(f"get_benchmark_queue received {res}")

        return res

    def upload_weight_file(self, bin_url: str) -> None:
        """Upload ONNX weight binary file.

        Args:
            bin_url (str): Url to upload ONNX weight

        Raises:
            FileNotFoundError: When bin file does not exists at given path.
        """
        if not os.path.exists(self.bin_path):
            log.error(
                f"Missing ONNX weight file at {self.bin_path}. You may need to retry exporting your model to ONNX "
                "using `owl.export`"
            )  # UX
            raise FileNotFoundError("ONNX bin file not found")

        upload_file_to_url(self.bin_path, bin_url)

    def poll_benchmark(self) -> None:
        """Poll for the benchmark result.

        Raises:
            ValueError: When unexpected signal is caught by SIGINT handler.
            RuntimeError: When error occurred during benchmarking process.
        """
        require_weight_upload = self.plan.paid

        def sigint_handler(sig: signal.Signals, frame: Any) -> None:
            if sig != signal.SIGINT:
                raise ValueError(f"Unexpected signals: {sig} (frame={frame})")
            print("")
            if require_weight_upload:
                log.warning(
                    "Escaping from the polling. The requested benchmark will fail as the weights are not uploaded"
                )  # UX
            else:
                log.info("Escaping from the polling. The benchmark will still run in the background")  # UX
            sys.exit(sig)

        if require_weight_upload:
            log.info(
                "Waiting for ONNX model weights to be uploaded. Please be patient. "
                "Benchmarking will fail if the weights are not uploaded"
            )  # UX

        original_sigint_handler = signal.signal(signal.SIGINT, sigint_handler)  # type: ignore

        count = 0
        error_log = ""
        status = BenchmarkStatus.IDLE
        informed = False
        while True:
            if count % 5 == 0:
                info = self.get_benchmark_queue()
                status = BenchmarkStatus(info.get("status", -999))

            if not require_weight_upload and not informed:
                log.info(
                    "Polling for benchmark result. Press Ctrl+C to escape. "
                    f"You can later find the results at {self.url}"
                )  # UX
                count = 0  # reset the count to start the owl from the beginning
                informed = True

            match status:
                case BenchmarkStatus.PRE_FETCHING | BenchmarkStatus.BENCHMARKING:
                    if require_weight_upload and len(bin_url := info.get("url", "")):
                        print()
                        self.upload_weight_file(bin_url)
                        require_weight_upload = False

                    message = "\r"
                    job_position = info.get("pos", None)
                    if job_position is not None:
                        dots = ". " * (count % 4)
                        spaces = "  " * (3 - (count % 4))
                        message = f"\rYour position in the queue: {job_position} {dots}{spaces}"  # UX

                    elif informed:
                        dots_before = "." * count
                        owl_emoji = "\U0001f989"
                        dots_after = "." * (19 - count)
                        message = f"\r[{dots_before}{owl_emoji}{dots_after}]"  # UX

                    print(f"{message}", end="", flush=True)

                case BenchmarkStatus.BENCHMARK_DONE:
                    print()
                    log.info("Benchmarking done")  # UX
                    break

                case _ if status.failed:
                    error_log = info.get("error_log", "")
                    break

            count = (count + 1) % 20
            time.sleep(2)

        signal.signal(signal.SIGINT, original_sigint_handler)
        if status.failed:
            status_message_dict = {
                BenchmarkStatus.FETCHING_ERR: "Benchmarking failed with pre-fetching.",
                BenchmarkStatus.TIMEOUT_ERR: "Benchmarking failed with timeout.",
                BenchmarkStatus.BENCHMARK_ERR: "Benchmarking failed during benchmark.",
                BenchmarkStatus.WEIGHT_GEN_ERR: "Benchmarking failed with weight generation.",
                BenchmarkStatus.STATUS_NOT_FOUND: "Benchmarking failed with an unexpected error.",
            }

            sep = "\n\t\t" if error_log else ""
            error_message = (
                f"{error_log}"
                f"{sep}{status_message_dict[status]} "
                f"{sep}Please try again, and if the problem persists, please report the issue at "
                f"{OWLITE_REPORT_URL} for further assistance"
            )

            print()
            log.error(error_message)
            raise RuntimeError("Benchmarking failed")

    def get_benchmark_result(self) -> BenchmarkResult:
        """Get the benchmarking result.

        Returns:
            BenchmarkResult: The information of an experiment if exists, None otherwise.

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

    def download_engine(self) -> None:
        """Download built engine.

        Raises:
            RuntimeError: When device is not set.
            HTTPError: When request was not successful.
        """
        try:
            resp = DEVICE_API_BASE.post(
                "/devices/trt",
                json={
                    "device_name": self.device.name,
                    "benchmark_key": self.benchmark_key,
                },
            )
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                log.error(
                    "Missing the engine to download. " "You may need to retry build the engine using `owl.benchmark`"
                    if self.plan.paid
                    else "The free plan doesn't support the engine download. "
                    "Upgrade to a higher plan to download the engine through OwLite with a seamless experience"
                )  # UX
            raise e
        assert isinstance(resp, dict)
        file_url = resp["trt_engine_url"]
        download_file_from_url(file_url, self.engine_path)

    def clear_engine(self) -> None:
        """Clear created the engine on device."""
        log.debug(f"Clear the engine on device: {self.device}, benchmark_key: {self.benchmark_key}")
        resp = DEVICE_API_BASE.post(
            "/devices/clear",
            json={
                "device_name": self.device.name,
                "benchmark_key": self.benchmark_key,
            },
        )
        assert isinstance(resp, str)
        if len(resp) > 0:
            log.debug(f"Clear the engine with url: {resp}")
            del_res = requests.delete(resp, timeout=OWLITE_API_DEFAULT_TIMEOUT)
            if not del_res.ok:
                del_res.raise_for_status()

    def log(self, message: str) -> None:
        """Log JSON-serialized metrics.

        Raises:
            HTTPError: When request was not successful.
        """
        resp = MAIN_API_BASE.post(
            "/projects/runs/update",
            json=self.payload(run_name=self.name, logs=message),
        )
        assert isinstance(resp, str)

    def payload(self, **kwargs: str | int) -> dict[str, str | int]:
        """Create payload for API requests.

        Raises:
            NotImplementedError: If not implemented by subclasses.
        """
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()
