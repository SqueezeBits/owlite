# pylint: disable=too-many-lines
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import torch
from packaging.version import Version
from torch.fx.graph_module import GraphModule
from torch.nn.parallel import DataParallel, DistributedDataParallel

from .api import Baseline, Experiment, Project
from .backend.fx.trace import symbolic_trace
from .compression import compress
from .core.cli.device import connect_to_first_available_device
from .core.constants import OWLITE_REPORT_URL, OWLITE_VERSION
from .core.device_settings import OWLITE_DEVICE_SETTINGS
from .core.github_utils import get_latest_version_from_github
from .core.logger import log
from .core.settings import OWLITE_SETTINGS
from .options import DynamicAxisOptions, DynamicInputOptions, ONNXExportOptions


@dataclass
class OwLite:
    """Class handling OwLite project, baseline, and experiment configurations.

    The OwLite class manages project, baseline, and experiment configurations within the OwLite system.
    It allows users to create or load projects, set baselines, create or duplicate experiments, convert models,
    and benchmark models against the specified configurations.
    """

    target: Baseline | Experiment
    module_args: tuple[Any, ...] | None = field(default=None)
    module_kwargs: dict[str, Any] | None = field(default=None)

    def convert(
        self,
        model: torch.nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> GraphModule:
        r"""Convert the model into a `torch.fx.GraphModule` object using the example input(s) provided.

        {% hint style="warning" %}

        The example input(s) provided for `owl.convert` will also be used by
        [`owl.export`](https://squeezebits.gitbook.io/owlite/python-api/owlite.owlite.owlite/owlite.owlite.export) for
        the ONNX and Engine conversion afterward. Therefore, it is crucial to provide appropriate example input(s)
        to ensure the correct behavior of your model.

        {% endhint %}

        Args:
            model (`torch.nn.Module`): The model to be compressed. Note that it must be an instance of
            `torch.nn.Module`, but not `torch.nn.DataParallel` or `torch.nn.DistributedDataParallel`. See
            [troubleshooting - Models wrapped with `torch.nn.DataParallel` or `torch.nn.parallel.DistributedDataParallel`](https://squeezebits.gitbook.io/owlite/troubleshooting/troubleshooting#models-wrapped-with-torch.nn.dataparallel-or-torch.nn.parallel.distributeddataparallel)
            for more details.
            *args: the example input(s) that would be passed to the model's forward method.
            **kwargs: the example input(s) that would be passed to the model's forward method.
            > These example inputs are required to convert the model into a
            [`torch.fx.GraphModule`](https://pytorch.org/docs/stable/fx.html)
            instance. Each input must be one of the following:
            * A `torch.Tensor` object
            * A tuple of `torch.Tensor` objects
            * A dictionary whose keys are strings and values are `torch.Tensor` objects.

        Returns:
            GraphModule: The `torch.fx.GraphModule` object converted from the `model`.

        Raises:
            HTTPError: When request for compression configuration was not successful.

        ### Behavior in each mode

        `owl.convert` behaves differently depending on the
        [mode](https://squeezebits.gitbook.io/owlite/python-api/owlite.owlite.init#two-different-modes-triggered-by-owlite.init)
        triggered by [`owlite.init`](https://squeezebits.gitbook.io/owlite/python-api/owlite.owlite.init).

        1. Baseline Mode: In this mode, `owl.convert` traces the input model with the example input(s).

        2. Experiment Mode: In this mode, the converted `torch.fx.GraphModule` object will be further modified
        according to the compression configuration from the experiment. This configuration could have been created by
        the user on the OwLite website, or copied from another experiment (in 'duplicate from' mode). If there's no
        compression configuration, it returns the same model as in baseline mode. For dynamic batch size baseline
        model without compression, create an experiment.

        Workflow:

        `owl.convert` goes through the following steps:

        1. Conversion: it converts the input model to the format configurable by OwLite, namely to a [torch.fx.GraphModule](https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule)
        instance, with the example input(s) provided via `*args` and `**kwargs`. This procedure might fail depending on
        your model's implementation and the [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)'s
        coverage in the PyTorch version you're using. If this is the case, you may need to find and fix the causes of
        the failure provided by the error message.

        2. Compression: In the experiment mode, it further compresses the converted model if the experiment's
        compression configuration exists. Keep in mind that you must setup the compression configuration via
        [OwLite Web UI](https://owlite.ai) before running the `owl.convert` in order to compress your model.

        Examples:
        **Baseline Mode**

        ```python
        import owlite
        import torch

        owl = owlite.init(project="testProject", baseline="sampleModel")

        # Create a sample model
        class SampleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 64, 3)
                self.pool1 = torch.nn.MaxPool2d(2, 2)
                self.conv2 = torch.nn.Conv2d(64, 128, 3)
                self.pool2 = torch.nn.MaxPool2d(2, 2)
                self.fc1 = torch.nn.Linear(128 * 7 * 7, 10)

        # Create a model instance
        model = SampleModel()

        # Convert the model
        model = owl.convert(model, torch.randn(4,3,64,64))

        # Print the model
        print(model)
        ```

        This code will create a sample model, convert it to a GraphModule in baseline mode, and export it to ONNX.
        The output of the code is as follows:

        ```
        OwLite [INFO] Connected device: NVIDIA RTX A6000
        OwLite [WARNING] Existing local directory found at /home/sqzb/workspace/owlite/testProject/sampleModel/sample
        Model. Continuing this code will overwrite the data
        OwLite [INFO] Created new project 'testProject'
        OwLite [INFO] Created new baseline 'sampleModel' at project 'testProject'
        OwLite [INFO] Converted the model
        GraphModule(
        (self_conv1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1))
        (self_pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (self_conv2): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
        (self_pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (self_fc1): Linear(in_features=6272, out_features=10, bias=True)
        )


        def forward(self, x : torch.Tensor):
            sqzb_module_device_canary = self.sqzb_module_device_canary
            getattr_1 = sqzb_module_device_canary.device;  sqzb_module_device_canary = None
            self_conv1 = self.self_conv1(x);  x = None
            relu = torch.nn.functional.relu(self_conv1);  self_conv1 = None
            self_pool1 = self.self_pool1(relu);  relu = None
            self_conv2 = self.self_conv2(self_pool1);  self_pool1 = None
            relu_1 = torch.nn.functional.relu(self_conv2);  self_conv2 = None
            self_pool2 = self.self_pool2(relu_1);  relu_1 = None
            view = self_pool2.view(-1, 6272);  self_pool2 = None
            self_fc1 = self.self_fc1(view);  view = None
            output_adapter = owlite_backend_fx_trace_output_adapter((self_fc1,));  self_fc1 = None
            return output_adapter
        ```

        **Experiment Mode**

        ```python
        import torch

        owl = owlite.init(project="testProject", baseline="sampleModel", experiment="conv")

        # Create a sample model
        class SampleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 64, 3)
                self.pool1 = torch.nn.MaxPool2d(2, 2)
                self.conv2 = torch.nn.Conv2d(64, 128, 3)
                self.pool2 = torch.nn.MaxPool2d(2, 2)
                self.fc1 = torch.nn.Linear(128 * 7 * 7, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = torch.nn.functional.relu(x)
                x = self.pool1(x)

                x = self.conv2(x)
                x = torch.nn.functional.relu(x)
                x = self.pool2(x)

                x = x.view(-1, 128 * 7 * 7)
                x = self.fc1(x)

                return x

        # Create a model instance
        model = SampleModel()

        # Convert the model
        model = owl.convert(model, torch.randn(4, 3, 64, 64))

        # Print the model
        print(model)
        ```

        This code will create a sample model, convert it to a GraphModule in experiment mode, and apply the compression
        configuration from the `init` function. The output of the code is as follows:

        ```
        OwLite [INFO] Connected device: NVIDIA RTX A6000
        OwLite [INFO] Experiment data will be saved in /home/sqzb/workspace/owlite/testProject/sampleModel/conv
        OwLite [INFO] Loaded existing project 'testProject'
        OwLite [INFO] Existing compression configuration for 'conv' found
        OwLite [INFO] Model conversion initiated
        OwLite [INFO] Compression configuration found for 'conv'
        OwLite [INFO] Applying compression configuration
        OwLite [INFO] Converted the model
        GraphModule(
        (self_conv1): QConv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1)
            (weight_quantizer): FakeQuantizer(ste(precision: 8, per_channel, quant_min: -127, quant_max: 127,
                            is_enabled: True, calib: AbsmaxCalibrator))
            (input_quantizer): FakeQuantizer(ste(precision: 8, per_tensor, quant_min: -128, quant_max: 127,
            q               zero_point: 0.0, is_zero_point_folded: False, is_enabled: True, calib: AbsmaxCalibrator))
        )
        (self_pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (self_conv2): QConv2d(
            64, 128, kernel_size=(3, 3), stride=(1, 1)
            (weight_quantizer): FakeQuantizer(ste(precision: 8, per_channel, quant_min: -127, quant_max: 127,
                            is_enabled: True, calib: AbsmaxCalibrator))
            (input_quantizer): FakeQuantizer(ste(precision: 8, per_tensor, quant_min: -128, quant_max: 127,
                            zero_point: 0.0, is_zero_point_folded: False, is_enabled: True, calib: AbsmaxCalibrator))
        )
        (self_pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (self_fc1): QLinear(
            in_features=6272, out_features=10, bias=True
            (weight_quantizer): FakeQuantizer(ste(precision: 8, per_channel, quant_min: -127, quant_max: 127,
                            is_enabled: True, calib: AbsmaxCalibrator))
            (input_quantizer): FakeQuantizer(ste(precision: 8, per_tensor, quant_min: -128, quant_max: 127,
                            zero_point: 0.0, is_zero_point_folded: False, is_enabled: True, calib: AbsmaxCalibrator))
        )
        (self_conv1_0_quantizer): FakeQuantizer(ste(precision: 8, per_tensor, quant_min: -128, quant_max: 127,
                            zero_point: 0.0, is_zero_point_folded: False, is_enabled: True, calib: AbsmaxCalibrator))
        (self_pool1_0_quantizer): FakeQuantizer(ste(precision: 8, per_tensor, quant_min: -128, quant_max: 127,
                            zero_point: 0.0, is_zero_point_folded: False, is_enabled: True, calib: AbsmaxCalibrator))
        (self_conv2_0_quantizer): FakeQuantizer(ste(precision: 8, per_tensor, quant_min: -128, quant_max: 127,
                            zero_point: 0.0, is_zero_point_folded: False, is_enabled: True, calib: AbsmaxCalibrator))
        (self_pool2_0_quantizer): FakeQuantizer(ste(precision: 8, per_tensor, quant_min: -128, quant_max: 127,
                            zero_point: 0.0, is_zero_point_folded: False, is_enabled: True, calib: AbsmaxCalibrator))
        (self_fc1_0_quantizer): FakeQuantizer(ste(precision: 8, per_tensor, quant_min: -128, quant_max: 127,
                            zero_point: 0.0, is_zero_point_folded: False, is_enabled: True, calib: AbsmaxCalibrator))
        )



        def forward(self, x : torch.Tensor):
            self_conv1_0_quantizer = self.self_conv1_0_quantizer(x);  x = None
            self_conv1 = self.self_conv1(self_conv1_0_quantizer);  self_conv1_0_quantizer = None
            relu = torch.nn.functional.relu(self_conv1);  self_conv1 = None
            self_pool1_0_quantizer = self.self_pool1_0_quantizer(relu);  relu = None
            self_pool1 = self.self_pool1(self_pool1_0_quantizer);  self_pool1_0_quantizer = None
            self_conv2_0_quantizer = self.self_conv2_0_quantizer(self_pool1);  self_pool1 = None
            self_conv2 = self.self_conv2(self_conv2_0_quantizer);  self_conv2_0_quantizer = None
            relu_1 = torch.nn.functional.relu(self_conv2);  self_conv2 = None
            self_pool2_0_quantizer = self.self_pool2_0_quantizer(relu_1);  relu_1 = None
            self_pool2 = self.self_pool2(self_pool2_0_quantizer);  self_pool2_0_quantizer = None
            view = self_pool2.view(-1, 6272);  self_pool2 = None
            self_fc1_0_quantizer = self.self_fc1_0_quantizer(view);  view = None
            self_fc1 = self.self_fc1(self_fc1_0_quantizer);  self_fc1_0_quantizer = None
            output_adapter = owlite_backend_fx_trace_output_adapter((self_fc1,));  self_fc1 = None
            return output_adapter
        ```
        """
        log.info("Converting the model")  # UX
        try:
            model = symbolic_trace(model, *args, **kwargs)
        except Exception as e:  # pylint: disable=broad-exception-caught
            log.error(
                "Failed to convert the model. This means that\n"
                "i) your model might have some codes that cannot be handled by `torch.compile`; or\n"
                "ii) the inputs provided for the model are incompatible with your model's 'forward' method.\n"
                "Check the full error message below and make changes accordingly. "
                f"Should the problem persist, please report the issue at {OWLITE_REPORT_URL} for further assistance"
            )  # UX
            raise e

        self.module_args = args
        self.module_kwargs = kwargs
        self.target.input_signature = model.meta["input_signature"]

        if isinstance(self.target, Experiment):
            if self.target.input_signature != self.target.baseline.input_signature:
                log.error(
                    "Input signature of current experiment does not match with baseline's. "
                    f"Please compare current input signature: {self.target.input_signature} "
                    f"and baseline input signature: {self.target.baseline.input_signature}"
                )  # UX
                raise RuntimeError("Input signature mismatch")

        if isinstance(self.target, Experiment) and self.target.has_config:
            model = compress(model, self.target.config)
            log.info("Applied compression configuration")  # UX

        return model

    @torch.no_grad()
    def export(
        self,
        model: GraphModule,
        onnx_export_options: ONNXExportOptions | None = None,
        dynamic_axis_options: DynamicAxisOptions | dict[str, int] | None = None,
    ) -> None:
        r"""Export the model converted by `owl.convert` to ONNX format.

        {% hint style=“warning" %}

        The ONNX model created by `owl.export` will also be used by
        [`owl.benchmark`](https://squeezebits.gitbook.io/owlite/python-api/owlite.owlite.owlite/owlite.owlite.benchmark)
        for the engine conversion afterward. Therefore, it is crucial to provide an appropriate pre-trained or
        calibrated model to ensure the correct behavior of your model.

        Generally, you can export any model with `owl.export` whether it is trained or not.
        However, keep in mind that some graph-level optimizations performed while building the engine
        depend on the values of your model's weight.

        For example, when you benchmark a quantized model without calibration, the `step_size` parameter of
        the fake quantizers in the model would be all initialized to zeros. These zero `step_size` values can make
        the behavior of the graph-level optimization different, leading to a different latency from a calibrated
        model's in the benchmarking stage.

        Therefore, we **strongly recommend**

        1. to export for benchmarking a pre-trained model in the baseline mode; and

        2. to perform either [PTQ calibration](https://squeezebits.gitbook.io/owlite/python-api/owlite.calibrators) or
        [QAT](https://squeezebits.gitbook.io/owlite/python-api/owlite.nn.functions) in experiment mode.

        {% endhint %}

        Args:
            model (`torch.fx.GraphModule`): The model converted by `owl.convert`. It must not be `torch.nn.DataParallel`
            or `torch.nn.DistributedDataParallel` instance.
            * See
            [troubleshooting - Models wrapped with `torch.nn.DataParallel` or `torch.nn.parallel.DistributedDataParallel`](https://squeezebits.gitbook.io/owlite/troubleshooting/troubleshooting#models-wrapped-with-torch.nn.dataparallel-or-torch.nn.parallel.distributeddataparallel)
            for more details.

            onnx_export_options (`owlite.ONNXExportOptions`, `optional`): Additional options for exporting ONNX.
            * OwLite exports your model into ONNX during the conversion using
            [`torch.onnx.export`](https://pytorch.org/docs/stable/onnx_torchscript.html#torch.onnx.export)
            behind the scenes. You can control some of the behaviors of `torch.onnx.export` by passing an
            `owlite.ONNXExportOptions` object to the `onnx_export_options` argument of `owlite.export`.
            Currently, you can only set `opset_version`, which defaults to 17. Other parameters of
            `torch.onnx.export` might be added in the future.

            dynamic_axis_options (`DynamicAxisOptions | dict[str, int]]`, `optional`): By default, the exported
            model will have the shapes of all input tensors set to match exactly those given when calling convert.
            To specify the axis of tensors as dynamic (i.e., known only at run-time), set `dynamic_axis_options`
            to a dictionary with schema:
            * KEY (`str`): the name of the input tensor.
            * VALUE (`int`): the axis to be dynamic.

        Raises:
            TypeError: When the `model` is an instance of `torch.nn.DataParallel` or `torch.nn.DistributedDataParallel`.
            RuntimeError: When `dynamic_axis_options` is set for baseline export.
            ValueError: When invalid `dynamic_axis_options` is given.

        ### Behavior in each mode

        `owl.export` behaves differently depending on the
        [mode](https://squeezebits.gitbook.io/owlite/python-api/owlite.owlite.init#two-different-modes-triggered-by-owlite.init)
        triggered by [`owlite.init`](https://squeezebits.gitbook.io/owlite/python-api/owlite.owlite.init).

        1. Baseline Mode: In this mode, `owl.export` traces the input model with the example input(s) and exports
        it to ONNX. Then, it sends the ONNX graph and the model to the server. This allows users to view the model
        graph on the web and apply compression.

        2. Experiment Mode: In this mode, `owl.export` exports the model after applying the compression
        configuration from the experiment or dynamic export options.

        Workflow:

        `owl.export` goes through the following steps:

        1. Exporting ONNX: It exports the input model into ONNX and saves it at your local workspace.
        In experiment mode, the model will be equipped with a dynamic axis if `dynamic_axis_options` was provided.

        2. Uploading ONNX: It then uploads the ONNX (without weights) to the OwLite server.

        Examples:
        **Baseline Mode**

        ```python
        import owlite

        # Initialize a baseline or experiment
        owl = owlite.init(...)

        # Initialize your model
        model = ...

        # Convert the model
        model = owl.convert(model, ...)

        # Export the model into ONNX
        model = owl.export(model)
        ```

        ```
        Checking 0/1...
        OwLite [INFO] Saving exported ONNX proto at
        /home/sqzb/workspace/owlite/testProject/sampleModel/testProject_sampleModel_sampleModel.onnx with external data
        testProject_sampleModel_sampleModel.bin
        OwLite [WARNING] External data file at
        /home/sqzb/workspace/owlite/testProject/sampleModel/testProject_sampleModel_sampleModel.bin will be overwritten.
        OwLite [INFO] Baseline ONNX saved at
        /home/sqzb/workspace/owlite/testProject/sampleModel/testProject_sampleModel_sampleModel.onnx
        OwLite [INFO] Uploaded the model excluding parameters
        ```

        **Experiment Mode with dynamic batch**

        ```python
        import owlite

        # Initialize a baseline or experiment
        owl = owlite.init(...)

        # Initialize your model
        model = ...

        # Convert the model
        model = owl.convert(model, ...)

        # Export the model into ONNX with dynamic axis options
        owl.export(model, dynamic_axis_options={"x": 0})
        ```

        ```
        Checking 0/1...
        OwLite [INFO] Saving exported ONNX proto at
        OwLite [INFO] `dynamic_axis_options` provided for the following inputs: 'x'
        /home/sqzb/workspace/owlite/testProject/sampleModel/dynamic/testProject_sampleModel_dynamic.onnx with external
        data testProject_sampleModel_dynamic.bin
        OwLite [INFO] Experiment ONNX saved at
        /home/sqzb/workspace/owlite/testProject/sampleModel/dynamic/testProject_sampleModel_dynamic.onnx
        OwLite [INFO] Uploading
        /home/sqzb/workspace/owlite/testProject/sampleModel/dynamic/testProject_sampleModel_dynamic.onnx
        100%|█████████████████████████████████████████████████████████████████████| 2.29k/2.29k [00:00<00:00, 123kiB/s]
        OwLite [INFO] Uploading done
        ```

        OwLite will create ONNX graph file and parameter file with the hierarchical structure below:

        ```
        - owlite
            - testProject
                - sampleModel
                    - dynamic
                        - testProject_SampleModel_dynamic.onnx
                        - testProject_SampleModel_dynamic.bin
        ```
        """
        if isinstance(model, DataParallel | DistributedDataParallel):
            model_type = f"torch.nn.parallel.{type(model).__name__}"
            log.error(
                f"{model_type} is not supported. Please use the attribute module "
                f"by unwrapping the model from {model_type}. Try owl.export(model.module)"
            )  # UX
            raise TypeError(f"{model_type} is not supported by export")
        if not isinstance(model, GraphModule):
            model_type = f"{type(model).__module__}.{type(model).__name__}"
            raise TypeError(f"Expected GraphModule, but got model of type {model_type}")

        if isinstance(dynamic_axis_options, dict):
            dynamic_axis_options = DynamicAxisOptions(dynamic_axis_options)
            keys_repr = ", ".join(f"'{key}'" for key in dynamic_axis_options.keys())
            log.info(f"`dynamic_axis_options` provided for the following inputs: {keys_repr}")  # UX

        if isinstance(self.target, Baseline):
            if dynamic_axis_options is not None:
                log.warning(
                    "The `dynamic_axis_options` provided for baseline will be ignored. "
                    "To export baseline model with dynamic input, "
                    "please create an experiment without compression configuration "
                    "and export it with `dynamic_axis_options`"
                )  # UX
            proto = self.target.export(
                model,
                self.module_args,
                self.module_kwargs,
                onnx_export_options=onnx_export_options,
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
            self.target.upload()

    def benchmark(
        self,
        dynamic_input_options: DynamicInputOptions | dict[str, dict[str, int]] | None = None,
        download_engine: bool = True,
    ) -> None:
        r"""Execute the benchmark for the converted model on a connected device.

        `owl.benchmark` uses the ONNX created by `owl.export`. The ONNX is sent to the connected device and converted
        to a engine compatible with the device's runtime, which is benchmarked behind the scenes. If the benchmark
        finishes successfully, the benchmark summary will be displayed on the terminal. The converted engine file will
        also be downloaded into the workspace.
        You can find more information about the benchmark results from the project page in
        [OwLite Web UI](https://owlite.ai/project).

        {% hint style="warning" %}

        In general, any model generated by `owl.export` can be benchmarked with `owl.benchmark`, regardless of
        whether it is trained or not. Additionally, the model to be benchmarked is already determined
        when `owl.export` is executed.

        To ensure accurate latency measurements, especially for quantized models, we strongly recommend using
        a pre-trained or calibrated model before using `owl.export`.

        For details on model preparation, please refer to the
        [PYTHON API/OwLite/Export](https://squeezebits.gitbook.io/owlite/python-api/owlite.owlite.owlite/owlite.owlite.export)

        {% endhint %}

        Args:
            dynamic_input_options (`dict[str, dict[str, int]]`): By default, the exported model will have the shapes
            of all input tensors set to exactly match those given when calling convert. To specify axes of tensors
            as dynamic (i.e. known only at run-time), set `dynamic_input_options` to a dictionary with schema:
            * KEY (`str`): the name of an input tensor.
            * VALUE (`dict[str, int]`): the dynamic range setting dictionary containing `"min"`, `"opt"`, `"max"`,
                `"test"` dimension size settings.
            download_engine (`bool`, optional): Whether to wait until the benchmarking is finished to download
                the engine. Defaults to True.

        Raises:
            TypeError: When the `model` is an instance of `torch.nn.DataParallel` or `torch.nn.DistributedDataParallel`.
            RuntimeError: When `dynamic_input_options` is set for baseline benchmark.
            ValueError: When invalid `dynamic_input_options` is given.

        Workflow:

        `owl.benchmark` goes through the following steps:

        1. Uploading The Model's Weights: It uploads the ONNX weight file to the device manager only for paid plan user.

        2. Creating A Runtime Engine: It converts the model into the runtime engine format (e.g. TensorRT engine)
        compatible with the device's runtime if necessary.

        3. Benchmarking on Device: It benchmarks the runtime engine on the device associated with the
        current baseline or experiment. When finished, it return the benchmarking results including latency, which will
        be displayed on the terminal.

        4. Downloading The Runtime Engine: The runtime engine file will be downloaded to the user's workspace. (paid
        plan only)

        Notes:
        **Benchmarking Considerations for Free Plan Users**
        >
        Benchmarking a model typically involves uploading its weight files for the most accurate results. However, if
        you're on the OwLite free plan, uploading weight files isn't currently supported. To address this, OwLite
        automatically generates random weights for your model's ONNX graph, allowing you to benchmark without needing
        your own weights. It's important to keep in mind that benchmarks using randomly generated weights might be less
        accurate compared to those using your actual model weights.
        >
        **Interrupting Benchmarking**
        >
        The benchmarking process can be interrupted at any time by pressing Ctrl+C. This will gracefully terminate
        the current experiment on your machine and display an exit message.

        * Early Interruption: If the interruption occurs before the model weights are uploaded, the benchmarking process
        on the device will also be aborted.
        * Late Interruption: If the interruption occurs after the model weights are uploaded, the benchmarking process
        will continue on the connected device. In either case, you'll be provided with a URL linking to the OwLite
        website for further project configuration.

        > Important Notes:
        The benchmark will still be accessible on the connected device after interruption, allowing you to resume the
        process later at your convenience. However, please be aware that manual engine retrieval will not be possible
        after interrupting the process.

        Examples:
        **Baseline Mode (or Experiment Mode with Static Batch Size)**

        ```python
        import owlite

        # Initialize a baseline or experiment
        owl = owlite.init(...)

        # Initialize your model
        model = ...

        # Convert the model
        model = owl.convert(model, ...)

        # Export the model into ONNX
        owl.export(model)

        # Benchmark the model
        owl.benchmark()
        ```

        **Experiment Mode with Dynamic Batch Size**

        ```python
        import owlite

        # Initialize a baseline or experiment
        owl = owlite.init(...)

        # Initialize your model
        model = ...

        # Convert the model
        model = owl.convert(model, ...)

        # Export the model into ONNX with dynamic axis options
        owl.export(model, dynamic_axis_options={"x": 0})

        # Benchmark the model with dynamic input options
        owl.benchmark(dynamic_input_options={"x": {"min": 1, "opt": 4, "max": 8, "test": 5}})
        ```

        ```
        OwLite [INFO] Benchmark initiated for the experiment 'dynamic' for the baseline 'sampleModel'
        in the project 'testProject'
        OwLite [INFO] Benchmark requested on NVIDIA RTX A6000 [TensorRT]
        OwLite [INFO] Polling for benchmark result. You are free to Ctrl+C away. When it is done, you can find the
        results at https://owlite.ai/project/detail/94af0e4c784fb1f
        Your position in the queue: 0
        OwLite [INFO] Uploading ONNX model weight to optimize the engine
        OwLite [INFO] Uploading /home/sqzb/workspace/owlite/testProject/sampleModel/dynamic/
        testProject_sampleModel_dynamic.bin
        100%|█████████████████████████████████████████████████████████████████████████████████| 541k/541k
        [00:00<00:00, 2.26MiB/s]
        OwLite [INFO] Uploading done
        [.........🦉..........]
        Benchmarking done
        OwLite [INFO] Experiment: dynamic
                    Latency: 0.0245361 (ms) on NVIDIA RTX A6000 [TensorRT]
                    For more details, visit https://owlite.ai/project/detail/94af0e4c784fb1f
        OwLite [INFO] Downloading file at
        /home/sqzb/workspace/owlite/testProject/sampleModel/dynamic/testProject_sampleModel_dynamic.engine
        100%|█████████████████████████████████████████████████████████████████████████████████| 554k/554k
        [00:00<00:00, 9.51MiB/s]
        OwLite [INFO] Downloading done
        ```

        OwLite will create the engine file with the hierarchical structure below.

        ```
        - owlite
        - testProject
            - SampleModel
            - dynamic
                - testProject_SampleModel_dynamic.onnx # created by owlite.export()
                - testProject_SampleModel_dynamic.bin # created by owlite.export()
                - testProject_SampleModel_dynamic.engine
        ```

        **Free plan user**

        However, please note that the Free plan does not allow you to export TensorRT engine files with the model's
        weights. Instead, a random weight engine will be created and you can only query its latency.
        You will not be able to get the generated engine.

        ```
        OwLite [INFO] Benchmark initiated for the experiment 'dynamic' for the baseline '"sampleModel"'
        in the project 'testProject'
        OwLite [INFO] Benchmark requested
        OwLite [INFO] Polling for benchmark result. You are free to Ctrl+C away. When it is done,
        you can find the results at https://owlite.ai/project/detail/94af0e4c784fb1f
        [.........🦉..........]
        Benchmarking done
        OwLite [INFO] Experiment: dynamic
                    Latency: 0.0327148 (ms) on NVIDIA RTX A6000
                    For more details, visit https://owlite.ai/project/detail/94af0e4c784fb1f
        OwLite [INFO] The free plan doesn't support the engine download. Upgrade to a higher plan to download
        the engine through OwLite with a seamless experience. Even so, OwLite still provides you ONNX
        so that you can generate a engine independently
        ```

        """
        if (
            isinstance(self.target, Experiment)
            and self.target.input_signature
            and self.target.input_signature.is_dynamic
        ):
            if dynamic_input_options is None:
                log.error(
                    "The `dynamic_input_options` for the experiment has `dynamic_input_options`. "
                    "Try `owl.benchmark(dynamic_input_options={...})`"
                )  # UX
                raise RuntimeError("Missing Dynamic input options")

            dynamic_input_options = DynamicInputOptions(dynamic_input_options)
            self.target.input_signature.fill_dynamic_ranges(dynamic_input_options)

        self.target.orchestrate_benchmark(download_engine=download_engine)

    def log(self, **kwargs: Any) -> None:
        """Record and send specific metrics to the server.

        These metrics can then be reviewed and analyzed on the web, along with other project data.
        This function can be used anytime after the initialization (`init`) step.

        Raises:
            TypeError: When data is not JSON serializable or not allowed logging.

        Usage:
        * The `log` function is used for logging metrics such as accuracy, loss, etc. for the model.
        * `owl.log` can take any number or string of keyword arguments, where each argument represents a different
        metric for the model.

        Example:
        ```python
        import owlite

        # Initialize a baseline or experiment
        owl = owlite.init(...)

        owl.log(accuracy=0.72, loss=1.2)
        ```


        Notes:
        * All arguments to the `log` function should be JSON serializable. If a provided argument is not serializable,
        a `TypeError` will be raised.
        * It's recommended to log your metrics near `owl.benchmark` call, as the state of the model at this point is
        closest to the deployed model. However, you can call the `log` function at any point after the `init` function
        is called, where the state of the model is expected to be the closest to the deployment.
        * You can update the logged metrics by calling the `log` function again with the new values.

        """
        if not all(isinstance(value, int | str | float) for value in kwargs.values()):
            log.error("Invalid value given to `owl.log`. The value for logging must be `int`, `str`, `float`")  # UX
            raise TypeError("Invalid value")
        try:
            self.target.log(json.dumps(kwargs))
        except TypeError as e:
            log.error("Invalid value given to `owl.log`. The metrics for logging must be JSON-serializable")  # UX
            raise e


# pylint: disable-next=too-many-branches, too-many-statements
def init(
    project: str,
    baseline: str,
    experiment: str | None = None,
    duplicate_from: str | None = None,
    description: str | None = None,
    device: str | None = None,
) -> OwLite:
    r"""Initialize the projects, baselines, and/or experiments.

    * A project comprises one or more baselines, the unmodified models you want to compress.
    * For each baseline in a project, you can create one or more experiments
    to benchmark various compression configurations for the baseline.
    * The project, baseline, or experiment name must only include alphanumeric characters
    and special characters among ()-_@:*&.

    ![Baseline-Experiment Hierarchy](https://github.com/SqueezeBits/owlite/assets/116608095/5bb3d540-4930-4f75-af84-6b4b609db392)

    Args:
        project (`str`): The new (or an existing) project name.
        baseline (`str`): The new baseline name.
        experiment (`str`. `optional`): The experiment name you want to create or load.
        If `experiment` is not provided, the process defaults to `baseline mode`; however,
        if `experiment` is specified, the process operates in `experiment mode`.
        duplicate_from (`str`. `optional`): The experiment name you want to clone.
        description (`str`. `optional`): A brief description of your project within 140 characters.
        (Required only for creating a new project.)
        device (`str`. `optional`): The name of the device to override the currently connected device.

    Raises:
        RuntimeError: When issues related to OwLite's operation are indicated.
        * If the current OwLite version is deprecated.
        * If not authenticated.
        * When the baseline is not found in the given project.
        * When the baseline and the connected device have different runtimes.
        * When the experiment to duplicate from is not found in the given baseline.

        ValueError: When provided values are invalid.
        * When the given baseline name is invalid.
        * When the given experiment name is invalid.
        * When the description length exceeds the limit.

    Returns:
        OwLite: An `owlite.OwLite` object configured for the designated project, baseline, and/or experiment.


    ### Two different modes triggered by `owlite.init`

    1. Baseline mode: Creating or loading a project and its baseline


    If you want to create a new project named "my_project" with a new baseline named "my_model",
    add the following line in your code:

    ```python
    owl = owlite.init(project="my_project", baseline="my_model")
    ```

    This function call can behave in different ways depending on the circumstances.

    * If the project named `"my_project"` already exists, the existing one will be loaded.
    * In contrast, if the baseline `"my_model"` already exists in the project `"my_project"`,
    it will still create a new baseline. The name of the newly created baseline will be renamed
    automatically by appending an appropriate postfix (e.g., `"my_model_1"` or `"my_model_2"`)


    2. Experiment mode: Creating or loading an experiment

    After creating a compression configuration at [owlite.ai](http://owlite.ai), you can benchmark the (compressed)
    model from your experiment as follows:

    ```python
    import owlite

    owl = owlite.init(project="my_project", baseline="my_model", experiment="my_experiment")
    ```

    This function call can behave in different ways depending on the circumstances.

    * If the experiment `"my_experiment"` is not found, OwLite will create a new one. In this case, the compression
    configuration for the newly created experiment will be empty. By calling
    [`owl.convert`](https://squeezebits.gitbook.io/owlite/python-api/owlite.owlite.owlite/owlite.owlite.benchmark) and
    [`owl.benchmark`](https://squeezebits.gitbook.io/owlite/python-api/owlite.owlite.owlite/owlite.owlite.benchmark)
    for this experiment, you can benchmark the baseline.
    * If the experiment `"my_experiment"` already exists, it downloads the compression configuration from the
    experiment. By calling
    [`owl.convert`](https://squeezebits.gitbook.io/owlite/python-api/owlite.owlite.owlite/owlite.owlite.convert) and
    [`owl.benchmark`](https://squeezebits.gitbook.io/owlite/python-api/owlite.owlite.owlite/owlite.owlite.benchmark),
    you can benchmark the compressed model from the experiment.

    Furthermore, you can clone an existing experiment by providing its name to `duplicate_from`.

    ```python
    import owlite

    owl = owlite.init(project="my_project", baseline="my_model", experiment="new_experiment",
    duplicate_from="existing_experiment")
    ```

    If `"new_experiment"` already exists, the newly created experiment will be renamed appropriately
    (e.g., `"new_experiment_1"` or `"new_experiment_2"`.)

    By performing these tasks, the `init` function ensures that the necessary setup is done for
    the project, baseline, and experiment within OwLite.

    Examples:
    1. Baseline Mode:

    ```python
    import owlite

    owl = owlite.init(project="testProject", baseline="sampleModel")
    ```

    This code creates a new project named `"testProject"` and a new baseline named `"sampleModel"` provided
    that the project with the same name does not already exist. `owlite.init` returns an `owlite.OwLite` object,
    which you will need for converting or benchmarking your baseline model.

    A typical output of this code is as follows:

    ```
    OwLite [INFO] Created a new project 'testProject'
    OwLite [INFO] Device connected: NVIDIA RTX A6000 [TensorRT]
    OwLite [INFO] Created a new baseline 'sampleModel' in the project 'testProject'
    OwLite [INFO] Experiment data will be saved in /home/sqzb/workspace/owlite/testProject/sampleModel
    ```

    2. Experiment Mode:

    ```python
    import owlite

    owl = owlite.init(project="testProject", baseline="sampleModel", experiment="conv")
    ```

    This code loads the experiment named `"conv"` for the baseline `"sampleModel"` in the project `"testProject"`.
    Likewise, `owlite.init` returns an `owlite.OwLite` object, which you will need for benchmarking the experiment.

    A typical output of this code is as follows:

    ```
    OwLite [INFO] Loaded the existing project 'testProject'
    OwLite [INFO] Device connected: NVIDIA RTX A6000 [TensorRT]
    OwLite [INFO] Loaded the existing experiment 'conv' for the baseline 'sampleModel' in the project 'testProject'
    OwLite [INFO] Compression configuration found for 'conv'
    OwLite [INFO] Experiment data will be saved in /home/sqzb/workspace/owlite/testProject/sampleModel/conv
    ```

    OwLite stores files, such as ONNX or TensorRT engine, generated from your code at
    `${OWLITE_HOME}/<project>/<baseline>/<experiment>`, where OWLITE_HOME is an environment variable
    that defaults to ` ./owlite `.

    3. Targeting A Device:
    If you want your baseline or experiment to target a different device from the one configured via OwLite CLI, use the
    `device` argument to specify a device of your choice. (Run `owlite device ls` to see the available devices.)

    For example, you may add the keyword argument `device="NVIDIA RTX A4000"` to create an experiment specifically
    targeting the device "NVIDIA RTX A4000".

    ```python
    import owlite

    owl = owlite.init(project="testProject", baseline="sampleModel", experiment="conv", device="NVIDIA RTX A4000")
    ```

    ```
    OwLite [INFO] Loaded the existing project 'testProject'
    OwLite [INFO] Device connected: NVIDIA RTX A4000 [TensorRT]
    OwLite [INFO] Loaded the existing experiment 'conv' for the baseline 'sampleModel' in the project 'testProject'
    OwLite [INFO] Compression configuration found for 'conv'
    OwLite [WARNING] Connected device not matching baseline 'sampleModel': NVIDIA RTX A6000 [TensorRT].
    Results may vary from latency breakdowns shown in the editor.
    OwLite [WARNING] Existing local directory found at /home/sqzb/workspace/owlite/testProject/sampleModel/conv.
    Continuing this code will overwrite the data
    ```


    Warnings:
    **Version mismatch**

    If your current version of OwLite is not the latest supported version, you'll receive the following warning message:

    ```
    OwLite [WARNING] A new version of OwLite is available. To ensure the best usage,
    please update the package to the latest version with the following command:
    pip install owlite --extra-index-url https://pypi.squeezebits.com/ --upgrade
    ```

    To ensure optimal performance and access to new features, it's recommended to update OwLite using the provided
    command.

    **Baseline device mismatch**

    If the device connected does not match the device associated with the baseline, you'll receive the following
    warning message:

    ```
    OwLite [WARNING] Connected device does not match baseline 'sampleModel': NVIDIA RTX A6000 [TensorRT].
    Results may vary from latency breakdowns shown in the editor.
    ```

    Ensure that the connected device matches the baseline device to obtain accurate latency breakdowns.

    **Experiment directory exists**

    When the local directory for your baseline or experiment already exists, OwLite will overwrite existing files.

    ```
    OwLite [WARNING] Existing local directory found at {target.home}. Continuing this code will overwrite the data.
    ```

    If you proceed, the existing data in the directory will be overwritten.

    The warning messages have been documented to provide insights into potential issues or actions that users may need
    to take into consideration.

    """
    owlite_latest_version = Version(get_latest_version_from_github())

    if OWLITE_VERSION.major < owlite_latest_version.major:
        log.error(
            f"Your current version ({OWLITE_VERSION}) is not supported. "
            "Please update the package to the latest version with the following command: "
            "pip install owlite --extra-index-url https://pypi.squeezebits.com/ --upgrade "
        )  # UX
        raise RuntimeError("Version is not supported")
    if OWLITE_VERSION < owlite_latest_version:
        log.warning(
            "A new version of OwLite is available. "
            "To ensure the best usage, please update the package to the latest version with the following command: "
            "pip install owlite --extra-index-url https://pypi.squeezebits.com/ --upgrade "
        )  # UX

    if OWLITE_SETTINGS.tokens is None:
        log.error("Please log in using 'owlite login'. Account not found on this device")  # UX
        raise RuntimeError("OwLite token not found")

    _validate_names(
        project=project,
        baseline=baseline,
        experiment=experiment,
        duplicate_from=duplicate_from,
    )
    if description and len(description) > 140:
        log.error(
            "The project description should consist of at most 140 characters. "
            "Note that the description is not required for loading an existing project"
        )  # UX
        raise ValueError("Description length exceeds limit")

    if experiment == baseline:
        log.error(
            f"Experiment name '{baseline}' is reserved for the baseline. Please try with a different experiment name"
        )  # UX
        raise ValueError("Invalid experiment name")

    proj: Project = Project.load_or_create(project, description=description)

    target: Baseline | Experiment
    if OWLITE_DEVICE_SETTINGS.connected is None:
        connect_to_first_available_device()
    assert OWLITE_DEVICE_SETTINGS.connected
    if device:
        target_device = OWLITE_DEVICE_SETTINGS.connected.manager.get(device)
    else:
        target_device = OWLITE_DEVICE_SETTINGS.connected
    log.info(f"Device connected: {target_device}")  # UX

    if experiment is None:
        if duplicate_from:
            log.warning(
                f"`duplicate_from='{duplicate_from}'` will be ignored as no value for `experiment` was provided"
            )  # UX
        target = Baseline.create(proj, baseline, target_device)
    else:
        existing_baseline = Baseline.load(proj, baseline, target_device)
        if existing_baseline is None:
            log.error(
                f"No such baseline: {baseline}. "
                f"Please check if the baseline name for the experiment '{experiment}' is correct"
            )  # UX
            raise RuntimeError("Baseline not found")
        if existing_baseline.device.runtime != target_device.runtime:
            log.error(
                f"Device runtime mismatched with the baseline's: {experiment}={target_device.runtime.name}, "
                f"{existing_baseline.name}={existing_baseline.device.runtime.name}"  # pylint: disable=no-member
                ". Please connect to a device with the same runtime or upload a compatible baseline"
            )  # UX
            raise RuntimeError("Runtime mismatch")
        if duplicate_from is None:
            target = Experiment.load_or_create(existing_baseline, experiment, target_device)
        else:
            existing_experiment = Experiment.load(existing_baseline, duplicate_from, target_device)
            if existing_experiment is None:
                log.error(
                    f"The experiment '{duplicate_from}' to duplicate from is not found. "
                    "Please check if the project name provided for `duplicate_from` argument is correct"
                )  # UX
                raise RuntimeError("Experiment not found")
            target = existing_experiment.clone(experiment)
        target.device = target_device

    if isinstance(target, Experiment) and target.device.name != target.baseline.device.name:
        log.warning(
            f"Connected device not matching baseline '{target.baseline.name}': "
            f"{target.baseline.device if target.baseline.device else 'UNKNOWN'}. "
            "Results may vary from latency breakdowns shown in the editor."
        )  # UX

    if os.path.exists(target.home):
        log.warning(
            f"Existing local directory found at {target.home}. Continuing this code will overwrite the data"
        )  # UX
    else:
        os.makedirs(target.home, exist_ok=True)
        log.info(f"Experiment data will be saved in {target.home}")  # UX

    return OwLite(target)


def _validate_names(**kwargs: Any) -> None:
    """Validate a list of names.

    Args:
        **kwargs: A dictionary where keys are identifiers and values are names to validate.

    Raises:
        ValueError: If any name is invalid.
    """
    invalid_keys = []
    regex = r"^[a-zA-Z0-9()\-_@:*&]+$"
    for key, name in kwargs.items():
        if name is None:
            continue
        if not re.fullmatch(regex, name):
            invalid_keys.append(key)
    if len(invalid_keys) > 0:
        invalid_items = ", ".join(f"{key}={kwargs[key]}" for key in invalid_keys)
        log.error(
            f"The following names do not meet the requirement: {invalid_items}. "
            "A valid name must consist of alphanumeric characters or special characters chosen from ()-_@:*&"
        )  # UX
        raise ValueError("Invalid name")
