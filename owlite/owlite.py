# type: ignore
"""OwLite Optimization Module

This module facilitates optimization and benchmarking of models using OwLite services."""
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch
from torch.fx import GraphModule  # type: ignore
from torch.nn.parallel import DataParallel, DistributedDataParallel

from owlite_core.cli.device import OWLITE_DEVICE_NAME
from owlite_core.constants import (
    OWLITE_FRONT_BASE_URL,
    OWLITE_REPO_PATH,
    OWLITE_REPORT_URL,
)
from owlite_core.owlite_settings import OWLITE_SETTINGS

from .api.device.devices import (
    download_trt_engine,
    poll_run_benchmark,
    request_trt_benchmark,
)
from .api.dove.doves import get_configuration, upload_baseline
from .api.main.baselines import check_baseline_existence, create_baseline
from .api.main.projects import create_or_load_project
from .api.main.runs import (
    copy_run,
    create_run,
    get_benchmark_key,
    get_run_info,
    update_run_info,
    upload_run_onnx_proto,
)
from .backend.fx.trace import symbolic_trace
from .backend.onnx.dynamize import configure_dynamic_dimensions
from .backend.onnx.export import export, get_input_shape_signature
from .logger import log
from .options import GraphQuantizationOptions, ONNXExportOptions
from .quantize import quantize


@dataclass
class OwLite:
    """Class handling OwLite project, baseline, and experiment configurations.

    The OwLite class manages project, baseline, and experiment configurations within the OwLite system.
    It allows users to create or load projects, set baselines, create or duplicate experiments, convert models,
    and benchmark models against the specified configurations.
    """

    project_id: str
    project_name: str
    baseline_name: str
    experiment_name: str
    onnx_export_options: ONNXExportOptions
    module_args: Optional[tuple[Any, ...]] = None
    module_kwargs: Optional[dict[str, Any]] = None

    @property
    def is_baseline(self) -> bool:  # pylint: disable=missing-function-docstring
        return self.baseline_name == self.experiment_name

    def convert(self, model: torch.nn.Module, *args, **kwargs) -> GraphModule:
        """Converts input model to compressed model.

        Args:
            model (torch.nn.Module): Model to compress.

        Returns:
            GraphModule: Compressed graph module.

        Raises:
            HTTPError: When request for compression configuration was not successful.
        """

        log.info("Model conversion initiated")
        try:
            model = symbolic_trace(model, *args, **kwargs)
        except Exception as e:  # pylint: disable=broad-exception-caught
            log.error(
                "Failed to extract the computation graph from the provided model. "
                "Please check the error message for details.\n"
                "If the issue persists, try replacing with a traceable node. "
                "In case the problem remain unresolved, kindly report it at "
                f"{OWLITE_REPORT_URL} for further assistance"
            )
            raise e

        self.module_args = args
        self.module_kwargs = kwargs

        if self.is_baseline:
            onnx_path = os.path.join(
                OWLITE_REPO_PATH,
                self.project_name,
                self.baseline_name,
                self.experiment_name,
                f"{self.project_name}_{self.baseline_name}_{self.experiment_name}.onnx",
            )
            export(
                model,
                (*self.module_args, self.module_kwargs),
                onnx_path,
                **asdict(self.onnx_export_options),
            )
            log.info("Baseline ONNX saved")
            upload_baseline(self.project_id, self.baseline_name, onnx_path, model)
            log.info("Uploaded the model excluding parameters")

        else:
            exp_info = get_run_info(self.project_id, self.baseline_name, self.experiment_name)
            assert exp_info is not None
            if not exp_info["config_id"]:
                log.warning("No compression configuration found, skipping the compression process")

            else:
                log.info(f"Compression configuration found for '{self.experiment_name}'")
                configuration_string = get_configuration(self.project_id, self.baseline_name, self.experiment_name)
                options = GraphQuantizationOptions.load(configuration_string)

                log.info("Applying compression configuration")
                model = quantize(model, options)

        log.info("Converted the model")
        return model

    def benchmark(
        self,
        model: GraphModule,
        dynamic_axes: Optional[dict[str, dict[int, dict[str, int]]]] = None,
    ) -> None:
        """Benchmarks given model.

        Args:
            model (GraphModule): Model to benchmark.
            dynamic_axes (Optional[dict[str, dict[int, dict[str, int]]]]):

                By default the exported model will have the shapes of all input tensors set to
                exactly match those given when calling convert. To specify axes of tensors as
                dynamic (i.e. known only at run-time), set `dynamic_axes` to a dict with schema:

                * KEY (str): an input name.

                * VALUE (dict[int, dict[str, int]]): a single item dictionary whose key is dynamic dimension of input
                    and value is a dynamic range setting dictionary containing min, opt, max, test dimension size
                    settings.

                For example::

                    import owlite

                    owl = owlite.init( ... )

                    class SumModule(torch.nn.Module):
                        def forward(self, x):
                            return torch.sum(x, dim=1)

                    model = owl.convert( ... )

                    ...

                    # set first(0-th) dimension of input x to be dynamic within the range of 1 ~ 8
                    # optimize for 4 and benchmark for 5
                    owl.benchmark(model, dynamic_axes={
                        "x": {
                            0: {
                                "min": 1,
                                "opt": 4,
                                "max": 8,
                                "test": 5,
                            }
                        }
                    })

        Raises:
            TypeError: When the `model` is an instance of `torch.nn.DataParallel` or `torch.nn.DistributedDataParallel`.
            RuntimeError: When `dynamic_axes` is set for baseline benchmark.
            ValueError: When invalid `dynamic_axes` is given.
        """
        if isinstance(model, (DataParallel, DistributedDataParallel)):
            _model_type = f"torch.nn.parallel.{type(model).__name__}"
            log.error(
                f"{_model_type} is not supported by benchmark, please use attribute module "
                f"to unwrap model from {_model_type}. Try owlite.benchmark(model.module)"
            )
            raise TypeError(f"{_model_type} is not supported by benchmark")

        if self.is_baseline:
            log.info(
                f"Benchmark initiated. '{self.baseline_name}' "
                "ONNX will be uploaded to the connected device for TensorRT execution and benchmark"
            )

            if dynamic_axes is not None:
                log.error(
                    "Baseline cannot be done with dynamic input. To benchmark baseline model with dynamic input, "
                    "please create a run without compression configuration and benchmark that run with dynamic input"
                )
                raise RuntimeError("Attempted dynamic baseline benchmark")

        else:
            log.info(
                f"Benchmark initiated. '{self.experiment_name}' "
                "ONNX will be created and uploaded to the connected device for TensorRT execution and benchmark"
            )

            dynamic_dimensions = None
            if dynamic_axes is not None:
                sep = "', '"
                log.info(f"dynamic_axes setting for following inputs are provided. '{sep.join(dynamic_axes.keys())}'")
                input_signature = get_input_shape_signature(
                    model, *(self.module_args or ()), **(self.module_kwargs or {})
                )
                dynamic_dimensions = configure_dynamic_dimensions(input_signature, dynamic_axes)

            onnx_path = os.path.join(
                OWLITE_REPO_PATH,
                self.project_name,
                self.baseline_name,
                self.experiment_name,
                f"{self.project_name}_{self.baseline_name}_{self.experiment_name}.onnx",
            )
            export(
                model,
                (*(self.module_args or ()), self.module_kwargs),
                onnx_path,
                **asdict(self.onnx_export_options),
                dynamic_dimensions=dynamic_dimensions,
            )
            log.info("Experiment ONNX saved")
            upload_run_onnx_proto(self.project_id, self.baseline_name, self.experiment_name, onnx_path, dynamic_axes)
            log.info("Uploaded the model excluding parameters")

        benchmark_key = get_benchmark_key(self.project_id, self.baseline_name, self.experiment_name)
        bin_path = os.path.join(
            OWLITE_REPO_PATH,
            self.project_name,
            self.baseline_name,
            self.experiment_name,
            f"{self.project_name}_{self.baseline_name}_{self.experiment_name}.bin",
        )
        request_trt_benchmark(benchmark_key, bin_path)
        log.info("TensorRT engine execution and benchmark successfully requested")

        poll_run_benchmark(self.project_id, benchmark_key)
        exp_info = get_run_info(self.project_id, self.baseline_name, self.experiment_name)

        assert exp_info is not None
        if self.is_baseline:
            log.info(
                "Latency\n"
                f"\t\tBaseline - {exp_info['latency']} on {exp_info['device_name']}\n"
                "\t\tConfigure the quantization settings located at "
                f"{OWLITE_FRONT_BASE_URL}/project/detail/{self.project_id}"
            )
        else:
            log.info(
                "Latency\n"
                f"\t\tConfigured - {exp_info['latency']} on {exp_info['device_name']}\n"
                "\t\tRetrieve the specifics of the experiment at "
                f"{OWLITE_FRONT_BASE_URL}/project/detail/{self.project_id}"
            )

        engine_path = os.path.join(
            OWLITE_REPO_PATH,
            self.project_name,
            self.baseline_name,
            self.experiment_name,
            f"{self.project_name}_{self.baseline_name}_{self.experiment_name}.engine",
        )
        download_trt_engine(benchmark_key, engine_path)

    def log(self, **kwargs) -> None:
        """Logs the model's metrics.

        Notes:
            Log metrics with OwLite like below

            ...

            owl = owlite.init(...)

            ...

            owl.log(accuracy=0.72, loss=1.2)

        Raises:
            TypeError: When data is not JSON serializable.
        """
        try:
            logs = json.dumps(kwargs)
        except TypeError as e:
            log.error("Data is not JSON serializable")
            raise e
        update_run_info(self.project_id, self.baseline_name, self.experiment_name, logs)


# pylint: disable-next=too-many-branches
def init(
    project: str,
    baseline: str,
    experiment: Optional[str] = None,
    duplicate_from: Optional[str] = None,
    description: str = "",
    onnx_export_options: Optional[ONNXExportOptions] = None,
) -> OwLite:
    """Sets project, baseline and experiment information in DB to proper state and creates `OwLite` instance.

    Args:
        project (str): OwLite project name.
        baseline (str): OwLite baseline name.
        experiment (str, optional): OwLite experiment name. Defaults to None.
        duplicate_from (str, optional): OwLite source experiment name. Defaults to None.
        description (str, optional): OwLite project description. Defaults to "".
        onnx_export_options (ONNXExportOptions, optional): Options for ONNX export. Defaults to None.

    Raises:
        RuntimeError: When not authenticated.
        ValueError: When invalid experiment name or baseline name is given.

    Returns:
        OwLite: Created `OwLite` instance.
    """

    if OWLITE_SETTINGS.tokens is None:
        log.error("Please log in using 'owlite login'. Account not found on this device")
        raise RuntimeError("OwLite token not found")

    if OWLITE_DEVICE_NAME is None:
        log.warning("Connected device not found. Please connect device by 'owlite device connect --name (name)'")
    else:
        log.info(f"Connected device: {OWLITE_DEVICE_NAME}")

    if experiment == baseline:
        log.error(f"Experiment name '{baseline}' is reserved for baseline. Please try with a different experiment name")
        raise ValueError("Invalid experiment name")

    dir_path = os.path.join(
        OWLITE_REPO_PATH,
        project,
        baseline,
        experiment or baseline,
    )
    if os.path.exists(dir_path):
        log.warning(f"Existing local directory found at {dir_path}. Continuing this code will overwrite the data")
    else:
        os.makedirs(dir_path, exist_ok=True)
        log.info(f"Experiment data will be saved in {dir_path}")

    # create or load project
    project_id = create_or_load_project(project, description)

    if experiment is None:
        if duplicate_from:
            log.warning(f"duplicate_from='{duplicate_from}' will be ignored as no value for experiment was provided")

        created_baseline = create_baseline(project_id, baseline)
        if created_baseline != baseline:
            log.warning(
                f"A baseline '{baseline}' already exists. "
                f"Created a new baseline '{created_baseline}' at project '{project}'"
            )
            baseline = created_baseline
        else:
            log.info(f"Created new baseline '{baseline}' at project '{project}'")

    else:
        if not check_baseline_existence(project_id, baseline):
            log.error(f"Baseline '{baseline}' not found. Please verify the entered baseline name and try again")
            raise ValueError("Invalid baseline name")

        if duplicate_from:
            experiment = copy_run(project_id, baseline, duplicate_from, experiment)
            log.info(
                f"Copied compression configuration from the experiment '{duplicate_from}' "
                f"to the new experiment '{experiment}'"
            )

        exp_info = get_run_info(project_id, baseline, experiment)
        if exp_info is None:
            create_run(project_id, baseline, experiment)
            log.info(f"Created a new experiment '{experiment}' in project '{project}'")
            exp_info = get_run_info(project_id, baseline, experiment)

        assert exp_info is not None
        if not exp_info["config_id"]:
            log.warning(f"Compression configuration for '{experiment}' not found")

        else:
            log.info(f"Existing compression configuration for '{experiment}' found")

    onnx_export_options = onnx_export_options or ONNXExportOptions()

    return OwLite(project_id, project, baseline, experiment or baseline, onnx_export_options)
