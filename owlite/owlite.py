import json
import os
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from torch.fx.graph_module import GraphModule
from torch.nn.parallel import DataParallel, DistributedDataParallel

from owlite_core.cli.device import OWLITE_DEVICE_NAME
from owlite_core.constants import OWLITE_REPORT_URL
from owlite_core.logger import log
from owlite_core.owlite_settings import OWLITE_SETTINGS

from .api import Baseline, Experiment, Project
from .backend.fx.trace import symbolic_trace
from .backend.onnx.signature import DynamicSignature, update_dynamic_signature
from .options import DynamicAxisOptions, DynamicInputOptions, ONNXExportOptions
from .quantize import quantize


@dataclass
class OwLite:
    """Class handling OwLite project, baseline, and experiment configurations.

    The OwLite class manages project, baseline, and experiment configurations within the OwLite system.
    It allows users to create or load projects, set baselines, create or duplicate experiments, convert models,
    and benchmark models against the specified configurations.
    """

    target: Union[Baseline, Experiment]
    module_args: Optional[tuple[Any, ...]] = None
    module_kwargs: Optional[dict[str, Any]] = None

    def convert(self, model: torch.nn.Module, *args: Any, **kwargs: Any) -> GraphModule:
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
                "Failed to convert the model. This means that\n"
                "i) your model might have some codes that cannot be handled by `torch.compile`; or\n"
                "ii) the inputs provided for the model are incompatible with your model's 'forward' method.\n"
                "Check the full error message below and make changes accordingly. "
                f"Should the problem persist, please report the issue at {OWLITE_REPORT_URL} for further assistance"
            )
            raise e

        self.module_args = args
        self.module_kwargs = kwargs

        if isinstance(self.target, Experiment) and self.target.has_config:
            model = quantize(model, self.target.config)
            log.info("Applied compression configuration")

        return model

    def export(
        self,
        model: GraphModule,
        onnx_export_options: Optional[ONNXExportOptions] = None,
        dynamic_axis_options: Optional[Union[DynamicAxisOptions, dict[str, dict[str, int]]]] = None,
    ) -> None:
        """Exports and uploads given model.

        Args:
            model (GraphModule): Model to export.
            onnx_export_options (Optional[ONNXExportOptions], optional): Options for ONNX export. Defaults to None.
            dynamic_axis_options (Optional[DynamicAxisOptions], optional):

                By default the exported model will have the shapes of all input tensors set to
                exactly match those given when calling convert. To specify axes of tensors as
                dynamic (i.e. known only at run-time), set `dynamic_axis_options` to a dict with schema:

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

                    # set first(0-th) dimension of input x
                    owl.export(
                        model,
                        dynamic_axis_options={
                            "x": {"axis": 0},
                        },
                    )

                    # or equivalently,
                    owl.export(
                        model,
                        dynamic_axis_options=owlite.DynamicAxisOptions(
                            {"x": owlite.DynamicAxisOption(axis=0)}
                        ),
                    )

        Raises:
            TypeError: When the `model` is an instance of `torch.nn.DataParallel` or `torch.nn.DistributedDataParallel`.
            RuntimeError: When `dynamic_axes` is set for baseline export.
            ValueError: When invalid `dynamic_axes` is given.
        """
        if isinstance(model, (DataParallel, DistributedDataParallel)):
            model_type = f"torch.nn.parallel.{type(model).__name__}"
            log.error(
                f"{model_type} is not supported. Please use the attribute module "
                f"by unwrapping the model from {model_type}. Try owl.export(model.module)"
            )
            raise TypeError(f"{model_type} is not supported by export")
        if not isinstance(model, GraphModule):
            model_type = f"{type(model).__module__}.{type(model).__name__}"
            raise TypeError(f"Expected GraphModule, but got model of type {model_type}")

        if isinstance(dynamic_axis_options, dict):
            dynamic_axis_options = DynamicAxisOptions(dynamic_axis_options)
            keys_repr = ", ".join(f"'{key}'" for key in dynamic_axis_options.keys())
            log.info(f"dynamic_axis_options provided for the following inputs: {keys_repr}")

        if isinstance(self.target, Baseline):
            if dynamic_axis_options is not None:
                log.warning(
                    "The `dynamic_axis_options` provided for baseline will be ignored. "
                    "To export baseline model with dynamic input, "
                    "please create an experiment without compression configuration "
                    "and export it with `dynamic_axes`"
                )
            proto = self.target.export(
                model, self.module_args, self.module_kwargs, onnx_export_options=onnx_export_options
            )
            self.target.upload(proto, model)
        else:
            proto = self.target.export(
                model,
                self.module_args,
                self.module_kwargs,
                dynamic_axis_options=dynamic_axis_options,
                onnx_export_options=onnx_export_options,
            )
            self.target.upload(
                proto,
                dynamic_axis_options=dynamic_axis_options,
            )

    def benchmark(
        self,
        dynamic_input_options: Optional[Union[DynamicInputOptions, dict[str, dict[str, int]]]] = None,
    ) -> None:
        """Benchmarks given model.

        Args:
            dynamic_input_options (Optional[Union[DynamicInputOptions, dict[str, dict[str, int]]]]):

                By default the exported model will have the shapes of all input tensors set to
                exactly match those given when calling convert. To specify axes of tensors as
                dynamic (i.e. known only at run-time), set `dynamic_axes` to a dict with schema:

                * KEY (str): an input name.

                * VALUE (dict[str, int]): a single item who is a dynamic range setting dictionary
                containing min, opt, max, test dimension size settings.

                For example::

                    import owlite

                    owl = owlite.init( ... )

                    class SumModule(torch.nn.Module):
                        def forward(self, x):
                            return torch.sum(x, dim=1)

                    model = owl.convert( ... )

                    ...

                    # set input x to be dynamic within the range of 1 ~ 8
                    # optimize for 4 and benchmark for 5
                    owl.benchmark(
                        model,
                        dynamic_input_options={
                            "x": {
                                "min": 1,
                                "opt": 4,
                                "max": 8,
                                "test": 5,
                            },
                        },
                    )

                    # or equivalently,
                    owl.benchmark(
                        model,
                        dynamic_input_options=owlite.DynamicInputOptions(
                            {"x": owlite.DynamicSizeOptions(min=1, opt=4, max=8, test=5)}
                        ),
                    )

        Raises:
            TypeError: When the `model` is an instance of `torch.nn.DataParallel` or `torch.nn.DistributedDataParallel`.
            RuntimeError: When `dynamic_axes` is set for baseline benchmark.
            ValueError: When invalid `dynamic_axes` is given.
        """

        if isinstance(self.target, Experiment) and isinstance(self.target.input_signature, DynamicSignature):
            if dynamic_input_options is None:
                log.error(
                    "The `dynamic_input_options` for the experiment has `dynamic_axis_options`. "
                    "Try owl.benchmark(dynamic_axis_options={...})"
                )
                raise RuntimeError("Dynamic options failed")
            if isinstance(dynamic_input_options, dict):
                dynamic_input_options = DynamicInputOptions(dynamic_input_options)
            self.target.input_signature = update_dynamic_signature(self.target.input_signature, dynamic_input_options)

        self.target.orchestrate_trt_benchmark()

    def log(self, **kwargs: Any) -> None:
        """Logs the model's metrics.

        Notes:
            Log metrics with OwLite like below

            ...

            owl = owlite.init(...)

            ...

            owl.log(accuracy=0.72, loss=1.2)

        Raises:
            TypeError: When data is not JSON serializable or not allowed logging.
        """
        if not all(isinstance(value, (int, str, float)) for value in kwargs.values()):
            log.error("Invalied value given to `owl.log`. The value for logging must be `int`, `str`, `float`")
            raise TypeError("Invalid value")
        try:
            self.target.log(json.dumps(kwargs))
        except TypeError as e:
            log.error("Invalid value given to `owl.log`. The metrics for logging must be JSON-serializable")
            raise e


# pylint: disable-next=too-many-branches
def init(
    project: str,
    baseline: str,
    experiment: Optional[str] = None,
    duplicate_from: Optional[str] = None,
    description: Optional[str] = None,
) -> OwLite:
    """Sets project, baseline and experiment information in DB to proper state and creates `OwLite` instance.

    Args:
        project (str): OwLite project name.
        baseline (str): OwLite baseline name.
        experiment (Optional[str], optional): OwLite experiment name. Defaults to None.
        duplicate_from (Optional[str], optional): OwLite source experiment name. Defaults to None.
        description (Optional[str], optional): OwLite project description. Defaults to None.

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
        log.warning("Connected device not found. Please connect to a device by 'owlite device connect --name (name)'")
    else:
        log.info(f"Connected device: {OWLITE_DEVICE_NAME}")

    if experiment == baseline:
        log.error(
            f"Experiment name '{baseline}' is reserved for the baseline. Please try with a different experiment name"
        )
        raise ValueError("Invalid experiment name")

    proj: Project = Project.load_or_create(project, description=description)

    target: Union[Baseline, Experiment]
    if experiment is None:
        if duplicate_from:
            log.warning(
                f"`duplicate_from='{duplicate_from}'` will be ignored as no value for `experiment` was provided"
            )
        target = Baseline.create(proj, baseline)
    else:
        existing_baseline = Baseline.load(proj, baseline)
        if existing_baseline is None:
            log.error(
                f"No such baseline: {baseline}. "
                f"Please check if the baseline name for the experiment '{experiment}' is correct"
            )
            raise ValueError("Invalid baseline name")
        if duplicate_from is None:
            target = Experiment.load_or_create(existing_baseline, experiment)
        else:
            existing_experiment = Experiment.load(existing_baseline, duplicate_from)
            if existing_experiment is None:
                log.error(
                    f"The experiment '{duplicate_from}' to duplicate from is not found. "
                    "Please check if the project name provided for `duplicate_from` argument is correct"
                )
                raise ValueError("Invalid experiment name")
            target = existing_experiment.clone(experiment)

    if os.path.exists(target.home):
        log.warning(f"Existing local directory found at {target.home}. Continuing this code will overwrite the data")
    else:
        os.makedirs(target.home, exist_ok=True)
        log.info(f"Experiment data will be saved in {target.home}")

    return OwLite(target)
