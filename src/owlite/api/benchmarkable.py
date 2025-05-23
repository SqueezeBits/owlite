# pylint: disable=duplicate-code, too-many-public-methods, too-many-statements
import os
import signal
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from functools import cached_property
from typing import Any

import onnx
import requests
import torch
from torch.fx.graph_module import GraphModule

from ..backend.config import ONNX_OPS_TO_SAVE_PARAMETERS_INTERNALLY
from ..backend.onnx.export import export
from ..backend.signature import Signature
from ..core.api_base import MAIN_API_BASE, NEST_API_BASE
from ..core.cache.device import Device
from ..core.cache.workspace import Workspace
from ..core.cli.api.login import whoami
from ..core.constants import OWLITE_API_DEFAULT_TIMEOUT, OWLITE_REPORT_URL, OWLITE_VERSION
from ..core.logger import log
from ..enums.benchmark_status import BenchmarkStatus
from ..enums.price_plan import PricePlan
from ..enums.runtime import Runtime
from ..options import DynamicAxisOptions, ONNXExportOptions
from .utils import download_file_from_url, upload_file_to_url


@dataclass
class BenchmarkResult:
    """Benchmark result."""

    name: str
    device_name: str
    latency: float
    vram: float


@dataclass
class Benchmarkable(ABC):
    """Base protocol for objects that can request a benchmark."""

    name: str
    device: Device

    @property
    @abstractmethod
    def workspace(self) -> Workspace:
        """The workspace of the current project."""

    @cached_property
    @abstractmethod
    def plan(self) -> PricePlan:
        """Pricing plan of current user."""

    @property
    @abstractmethod
    def input_signature(self) -> Signature | None:
        """Input signature of model."""

    @property
    @abstractmethod
    def url(self) -> str:
        """The URL to the relevant page for this object."""

    @property
    @abstractmethod
    def home(self) -> str:
        """The directory path for writing outputs produced by this object."""

    @property
    @abstractmethod
    def label(self) -> str:
        """A unique label for this object."""

    @property
    @abstractmethod
    def skipped_optimizers(self) -> list[str] | None:
        """The list of optimizers to be skipped."""

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
        """The file path for writing the result."""
        return os.path.join(self.home, f"{self.label}.{self.device.runtime.file_ext}")

    @property
    def version_payload(self) -> dict[str, str]:
        """Current environment payload."""
        return {"owlite_version": str(OWLITE_VERSION), "torch_version": str(torch.__version__)}

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
        *,
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
        print(f"skipped_optimizers: {self.skipped_optimizers}")
        export(
            model,
            (*args, kwargs),
            self.onnx_path,
            self.bin_path,
            dynamic_axis_options=dynamic_axis_options,
            ops_to_save_parameter_internally=(
                ONNX_OPS_TO_SAVE_PARAMETERS_INTERNALLY if self.device.runtime == Runtime.QNN else []
            ),
            skipped_optimizers=self.skipped_optimizers,
            **(onnx_export_options.model_dump()),
        )
        log.info(f"{type(self).__name__} ONNX saved at {self.onnx_path}")  # UX
        return onnx.load(self.onnx_path, load_external_data=False)

    @abstractmethod
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

    def orchestrate_benchmark(self, download_engine: bool = True) -> None:
        """Orchestrate the end-to-end benchmark pipeline.

        Args:
            download_engine (bool, optional): Whether to wait until the benchmarking is finished to download
                the engine. Defaults to True.
        """
        if self.device is None:
            log.warning(
                "Cannot initiate benchmark. Please connect to a device first "
                "using 'owlite device connect --name (name)'"
            )  # UX
            return

        log.info(f"Benchmark initiated for the {self}")  # UX
        self.request_benchmark()
        log.info(f"Benchmark requested on '{self.device}'")  # UX

        self.poll_benchmark(wait_for_the_results=download_engine)

        if download_engine:
            result = self.get_benchmark_result()
            indent = " " * 14
            log.info(
                f"{type(self).__name__}: {result.name}\n"
                f"{indent}Latency: {result.latency} (ms) on {result.device_name}\n"
                f"{indent}For more details, visit {self.url}"
            )  # UX
            self.download_engine()
            self.clear_engine()

    def request_benchmark(self) -> None:
        """Request benchmark.

        Raises:
            ValueError: When device is not set.
            HTTPError: When request was not successful.
        """
        resp = NEST_API_BASE.post("/devices/jobs/assign", json=self.benchmark_payload())
        assert isinstance(resp, str)
        self.log_benchmark_priority()
        log.debug(f"request_benchmark received {resp}")

    def get_benchmark_queue(self) -> dict:
        """Get queueing information of an experiment.

        Returns:
            dict: Queueing information of an experiment.

        Raises:
            HTTPError: When request was not successful.
        """
        res = NEST_API_BASE.post("/devices/jobs/queue", json=self.benchmark_payload())
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

    def abort_benchmark(self) -> None:
        """Abort the requested benchmark on the device.

        Raises:
            HTTPError: When request was not successful.
        """
        res = NEST_API_BASE.post(
            "/devices/jobs/abort",
            json=self.benchmark_payload(),
        )
        assert isinstance(res, str)
        log.debug(f"abort_benchmark received {res}")

    def poll_benchmark(self, wait_for_the_results: bool = True) -> None:
        """Poll for the benchmark result.

        Args:
            wait_for_the_results (bool, optional): Whether to wait for the benchmark results.
                If False, the polling will be finished as soon as the weights are uploaded. Defaults to True.

        Raises:
            ValueError: When unexpected signal is caught by SIGINT handler.
            RuntimeError: When error occurred during benchmarking process.
        """

        def sigint_handler(sig: signal.Signals, frame: Any) -> None:
            if sig != signal.SIGINT:
                raise ValueError(f"Unexpected signals: {sig} (frame={frame})")
            print()
            if benchmark_status == BenchmarkStatus.PRE_FETCHING:
                self.abort_benchmark()
            else:
                log.info("Escaping from the polling. The benchmark will still run in the background")  # UX
            sys.exit(sig)

        iteration_count = 0
        error_log = ""
        benchmark_status = BenchmarkStatus.IDLE
        weight_file_uploaded = False

        original_sigint_handler = signal.signal(signal.SIGINT, sigint_handler)  # type: ignore
        log.info(
            "Waiting for benchmark results. Press Ctrl+C to escape from the polling. "
            "If you escape before the engine is built, the requested benchmark will be aborted."
        )  # UX

        while True:
            if iteration_count % 5 == 0:
                benchmark_info = self.get_benchmark_queue()
                benchmark_status = BenchmarkStatus(benchmark_info.get("status", -999))

            if len(bin_url := benchmark_info.get("url", "")) and not weight_file_uploaded:
                print()
                self.upload_weight_file(bin_url)
                weight_file_uploaded = True

                if not wait_for_the_results:
                    log.info(
                        f"The benchmark is running on the device remotely. You can later find the results at {self.url}"
                    )  # UX
                    break
                log.info(
                    "Polling for benchmark result. Press Ctrl+C to escape. "
                    f"You can later find the results at {self.url}"
                )  # UX

            message = ""
            match benchmark_status:
                case BenchmarkStatus.PRE_FETCHING | BenchmarkStatus.UPLOADING:
                    queue_position = benchmark_info.get("pos", None)
                    progress_dots = ". " * (iteration_count % 4)
                    message = (
                        f"\rYour position in the queue: {queue_position} {progress_dots}"
                        if queue_position is not None
                        else f"\rWaiting for the file fetch {progress_dots}"
                    )

                case BenchmarkStatus.BENCHMARKING:
                    dots_before = "." * iteration_count
                    owl_emoji = "\U0001f989"
                    dots_after = "." * (19 - iteration_count)
                    message = f"\r[{dots_before}{owl_emoji}{dots_after}]"  # UX

                case BenchmarkStatus.BENCHMARK_DONE:
                    print()
                    log.info("Benchmarking done")  # UX
                    break

                case _ if benchmark_status.failed:
                    error_log = benchmark_info.get("error_log", "")
                    break

            print(f"{message:40}", end="", flush=True)
            iteration_count = (iteration_count + 1) % 20
            time.sleep(2)

        signal.signal(signal.SIGINT, original_sigint_handler)
        if benchmark_status.failed:
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
                f"{sep}{status_message_dict[benchmark_status]} "
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
            resp = NEST_API_BASE.post("/devices/trt", json=self.benchmark_payload())
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                log.error(
                    "Missing the engine to download. You may need to retry build the engine using `owl.benchmark`"
                )  # UX
            raise e
        assert isinstance(resp, dict)
        file_url = resp["trt_engine_url"]
        download_file_from_url(file_url, self.engine_path)

    def clear_engine(self) -> None:
        """Clear created the engine on device."""
        log.debug(f"Clear the engine on device: {self.device}, benchmark_key: {self.benchmark_key}")
        resp = NEST_API_BASE.post("/devices/clear", json=self.benchmark_payload())
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
        resp = MAIN_API_BASE.post("/projects/runs/update", json=self.payload(run_name=self.name, logs=message))
        assert isinstance(resp, str)

    def log_benchmark_priority(self) -> None:
        """Log benchmark priority information for free plan users.

        Logs information about the remaining benchmark quota for free plan users.
        For paid plan users, this method returns without doing anything.
        """
        if self.plan.paid:
            return
        priority_queues_count = whoami().priority_queues_count
        if priority_queues_count > 20:
            log.info(
                "You have used all 20 Priority Queue Benchmarks available for all Free Plan workspaces. "
                "Benchmark will now proceed using the Standard Queue. Usage resets on the 1st of each month."
            )  # UX
        else:
            log.info(
                "In Free Plan Workspaces, requested benchmarks are processed through the priority queue, "
                "up to 20 per month. Exceeding this limit may result in a lower priority in the Benchmark queue."
            )  # UX
            log.info(f"Remaining quota: {20 - priority_queues_count}/20")  # UX

    @abstractmethod
    def payload(self, **kwargs: str | int) -> dict[str, str | int]:
        """Create payload for API requests."""

    def benchmark_payload(self, **kwargs: str | int) -> dict[str, str | int]:
        """Create benchmark payload for API requests."""
        p: dict[str, str | int] = {
            "workspace_id": self.workspace.id,
            "device_name": self.device.name,
            "benchmark_key": self.benchmark_key,
        }
        if self.device.runtime == Runtime.QNN:
            assert self.device.runtime_extra
            p["qualcomm_device_name"] = self.device.runtime_extra
        p.update(kwargs)
        return p
