"""An alternative for torch.onnx.export with extra optimizations."""

# pylint: disable=protected-access
import io
import os
import tempfile
from collections.abc import Collection, Mapping
from typing import Any

import onnx
import torch
import torch.onnx.errors
from onnx import ModelProto
from onnx.shape_inference import infer_shapes, infer_shapes_path
from torch.fx.graph_module import GraphModule

from ...core.constants import OWLITE_VERSION
from ...core.logger import log
from ...enums import ModelStatus
from ...nn import FakeFPQuantizer, FakeQuantizer
from ...nn.functions import clq_function
from ...options import DynamicAxisOptions
from ..config import STRICT_ONNX_FUNCTIONALITY_CHECKING, STRICT_ONNX_SHAPE_INFERENCE
from ..fx.filter_warnings import FilterWarningsCausedByPasses
from ..fx.transforms import clip_narrow_range_weights, fuse_bn
from ..signature import Signature
from ..utils import (
    ATOL_FP16,
    RTOL_FP16,
    nodestr,
)
from .dynamize import dynamize
from .export_with_external_data import export_with_external_data
from .model_checking import compare
from .optimize import optimize, optimize_path


# pylint: disable-next=too-many-arguments, too-many-locals, invalid-name, too-many-positional-arguments
def export(
    module: torch.nn.Module,
    args: tuple[Any, ...] | torch.Tensor,
    f: str,
    export_params: bool = True,
    verbose: bool = False,
    training: torch._C._onnx.TrainingMode = torch._C._onnx.TrainingMode.EVAL,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    operator_export_type: torch._C._onnx.OperatorExportTypes = torch._C._onnx.OperatorExportTypes.ONNX,
    opset_version: int = 17,
    do_constant_folding: bool = True,
    keep_initializers_as_inputs: bool | None = None,
    custom_opsets: Mapping[str, int] | None = None,
    export_modules_as_functions: bool | Collection[type[torch.nn.Module]] = False,
    export_in_tempdir: bool = False,
    skip_optimization: bool = False,
    check_n: int = 1,
    skip_fuse_bn: bool = False,
    skipped_optimizers: list[str] | None = None,
    dynamic_axis_options: DynamicAxisOptions | None = None,
    ops_to_save_parameter_internally: list[tuple[str, list[int]] | str] | None = None,
) -> None:
    r"""Export a module into ONNX format.

    Args:
        module (torch.nn.Module): The model to be exported.
        args (tuple[Any, ...] | torch.Tensor): Argument of a `module`.

            args can be structured either as:

            1. ONLY A TUPLE OF ARGUMENTS::

                args = (x, y, z)

            The tuple should contain model inputs such that `module(*args)` is a valid
            invocation of the model. Any non-Tensor arguments will be hard-coded into the
            exported model; any Tensor arguments will become inputs of the exported model,
            in the order they occur in the tuple.

            2. A TENSOR::

                args = torch.Tensor([1])

            This is equivalent to a 1-ary tuple of that Tensor.

            3. A TUPLE OF ARGUMENTS ENDING WITH A DICTIONARY OF NAMED ARGUMENTS::

                args = (
                    x,
                    {
                        "y": input_y,
                        "z": input_z
                    }
                )

            All but the last element of the tuple will be passed as non-keyword arguments,
            and named arguments will be set from the last element. If a named argument is
            not present in the dictionary, it is assigned the default value, or None if a
            default value is not provided.

            .. note::
                If a dictionary is the last element of the args tuple, it will be
                interpreted as containing named arguments. In order to pass a dict as the
                last non-keyword arg, provide an empty dict as the last element of the args
                tuple. For example, instead of::

                    export(
                        module,
                        (
                            x,
                            # WRONG: will be interpreted as named arguments
                            {y: z}
                        ),
                        "test.onnx.pb"
                    )

                Write::

                    export(
                        module,
                        (
                            x,
                            {y: z},
                            {}
                        ),
                        "test.onnx.pb"
                    )
        f (str): A string containing a file name. A binary protocol buffer will be written to this file.
        export_params (bool, optional): If True, all parameters will
            be exported. Set this to False if you want to export an untrained model.
            In this case, the exported model will first take all of its parameters
            as arguments, with the ordering as specified by `module.state_dict().values()`. Defaults to True.
        verbose (bool, optional): If True, prints a description of the
            model being exported to stdout. In addition, the final ONNX graph will include the
            field `doc_string` from the exported model which mentions the source code locations
            for `module`. If True, ONNX exporter logging will be turned on. Defaults to False.
        training (torch._C._onnx.TrainingMode, optional): Defaults to torch._C._onnx.TrainingMode.EVAL.
            * `TrainingMode.EVAL`: export the model in inference mode.
            * `TrainingMode.PRESERVE`: export the model in inference mode if model.training is
                False and in training mode if model.training is True.
            * `TrainingMode.TRAINING`: export the model in training mode. Disables optimizations
                which might interfere with training.
        input_names (list[str] | None, optional): Names to assign to the input nodes of the graph, in order.
            Names of `module.forward` arguments will be used when None is given. Defaults to None.
        output_names (list[str] | None, optional): Names to assign to the output nodes of the graph, in order.
            Defaults to None.
        operator_export_type (torch._C._onnx.OperatorExportTypes, optional):
            Defaults to `torch._C._onnx.OperatorExportTypes.ONNX`.
            * `OperatorExportTypes.ONNX`: Export all ops as regular ONNX ops (in the default opset domain).
            * `OperatorExportTypes.ONNX_FALLTHROUGH`: Try to convert all ops
                to standard ONNX ops in the default opset domain. If unable to do so
                (e.g. because support has not been added to convert a particular torch op to ONNX),
                fall back to exporting the op into a custom opset domain without conversion. Applies
                to `custom ops <https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html>`_
                as well as ATen ops. For the exported model to be usable, the runtime must support
                these non-standard ops.
            * `OperatorExportTypes.ONNX_ATEN`: All ATen ops (in the TorchScript namespace "aten")
                are exported as ATen ops (in opset domain "org.pytorch.aten").
                `ATen <https://pytorch.org/cppdocs/#aten>`_ is PyTorch's built-in tensor library, so
                this instructs the runtime to use PyTorch's implementation of these ops.

                .. warning::

                    Models exported this way are probably runnable only by Caffe2.

                    This may be useful if the numeric differences in implementations of operators are
                    causing large differences in behavior between PyTorch and Caffe2 (which is more
                    common on untrained models).
            * `OperatorExportTypes.ONNX_ATEN_FALLBACK`: Try to export each ATen op
                (in the TorchScript namespace "aten") as a regular ONNX op. If we are unable to do so
                (e.g. because support has not been added to convert a particular torch op to ONNX),
                fall back to exporting an ATen op. See documentation on OperatorExportTypes.ONNX_ATEN for
                context.
                For example::

                    graph(%0 : Float):
                    %3 : int = prim::Constant[value=0]()
                    # conversion unsupported
                    %4 : Float = aten::triu(%0, %3)
                    # conversion supported
                    %5 : Float = aten::mul(%4, %0)
                    return (%5)

                Assuming `aten::triu` is not supported in ONNX, this will be exported as::

                    graph(%0 : Float):
                    %1 : Long() = onnx::Constant[value={0}]()
                    # not converted
                    %2 : Float = aten::ATen[operator="triu"](%0, %1)
                    # converted
                    %3 : Float = onnx::Mul(%2, %0)
                    return (%3)

                If PyTorch was built with Caffe2 (i.e. with `BUILD_CAFFE2=1`), then
                Caffe2-specific behavior will be enabled, including special support
                for ops are produced by the modules described in
                `Quantization <https://pytorch.org/docs/stable/quantization.html>`_.

                .. warning::

                    Models exported this way are probably runnable only by Caffe2.
        opset_version (int, optional): The version of the default (ai.onnx) opset
            <https://github.com/onnx/onnx/blob/master/docs/Operators.md> to target. Must be >= 7 and <= 18.
            Defaults to 17.
        do_constant_folding (bool, optional): Apply the constant-folding optimization.
            Constant-folding will replace some of the ops that have all constant inputs
            with pre-computed constant nodes. Defaults to True.
        keep_initializers_as_inputs (bool | None, optional): If True, all the initializers
            (typically corresponding to parameters) in the exported graph will also be added
            as inputs to the graph. If False, then initializers are not added as inputs to the
            graph, and only the non-parameter inputs are added as inputs. This may allow for
            better optimizations (e.g. constant folding) by backends/runtimes. Defaults to None.
        custom_opsets (Mapping[str, int] | None, optional): A dict with schema:

            * KEY (str): opset domain name
            * VALUE (int): opset version

            If a custom opset is referenced by ``model`` but not mentioned in this dictionary,
            the opset version is set to 1. Only custom opset domain name and version should be
            indicated through this argument. Defaults to None.
        export_modules_as_functions (bool | Collection[type[torch.nn.Module]], optional): Flag to enable
            exporting all ``nn.Module`` forward calls as local functions in ONNX. Or a set to indicate the
            particular types of modules to export as local functions in ONNX.
            This feature requires ``opset_version`` >= 15, otherwise the export will fail. This is because
            ``opset_version`` < 15 implies IR version < 8, which means no local function support.
            Module variables will be exported as function attributes. There are two categories of function
            attributes. Defaults to False.
        export_in_tempdir (bool, optional): If True, the exporting process will be performed in a temporary directory
            instead of in the CPU memory. If `module` has total parameter size larger than 2GB, this flag will be
            automatically set to True. Defaults to False.
        skip_optimization (bool, optional): If False, the exported ONNX model proto will be optimized.
            If True, the optimization will be skipped. However, turning this flag to False is for debugging purpose only
            and might cause unexpected behavior. Defaults to False.
        check_n (int, optional): Only available when `skip_optimization=False`. The number of times to check
            functionality of the optimized ONNX proto against the original one with randomly generated inputs.
            Defaults to 1.
        skip_fuse_bn (bool, optional): Only available when `skip_optimization=False`. Whether to skip batchnorm-fusion.
            Defaults to False.
        skipped_optimizers (list[str] | None, optional): Only available when `skip_optimization=False`. The list of
            onnx-simplifier passes to skip. Defaults to None.
            See https://github.com/onnx/optimizer/tree/master/onnxoptimizer/passes for available passes.
        dynamic_axis_options (DynamicAxisOptions | None, optional): A `DynamicAxisOptions` object indicating
            which input tensor(s) should be configured with a dynamic axis.
            Defaults to None.
        ops_to_save_parameter_internally (list[tuple[str, list[int]] | str] | None, optional): (deprecated) ONNX
            operation types to store parameter within the ONNX file. Defaults to None.

    Raises:
        TypeError: If `f` is not a string.
        ValueError: If the quantizer has invalid condition.
        `torch.onnx.errors.CheckerError`: If the ONNX checker detects an invalid ONNX graph.
        `torch.onnx.errors.UnsupportedOperatorError`: If the ONNX graph cannot be exported because it
            uses an operator that is not supported by the exporter.
        `torch.onnx.errors.OnnxExporterError`: Other errors that can occur during export.
            All errors are subclasses of :class:`errors.OnnxExporterError`.
    """
    if not isinstance(f, str):
        raise TypeError("owlite.onnx.export requires the argument `f` to be a string.")

    if ops_to_save_parameter_internally and len(ops_to_save_parameter_internally):
        log.warning("Saving parameters internally might cause unexpected error as it is deprecated since v2.2.0")

    if module.training:
        log.warning("Exporting a module in training mode to ONNX might corrupt the functionality")  # UX

    if isinstance(module, GraphModule):
        if module.meta.get("status") == ModelStatus.COMPRESSED:
            log.warning(
                "This module has not yet been calibrated. "
                "The onnx that comes out of this module may have unexpected results in accuracy and latency."
            )
        if opset_version < 19 and any(isinstance(submodule, FakeFPQuantizer) for submodule in module.modules()):
            raise ValueError(f"FP quantizers are supported from opset 19. But got {opset_version}")
        if opset_version < 19 and any(
            isinstance(submodule, FakeQuantizer) and submodule.step_size.dtype != torch.float
            for submodule in module.modules()
        ):
            raise ValueError(
                "The element type of the input `x` of ONNX QuantizeLinear can be 'bfloat16' or 'float16' "
                f"since opset 19. Please bump up the {opset_version} to 19 or higher. "
                "(See https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html for more details.)"
            )
        clip_narrow_range_weights(module)
        # Batch Norm Fusing
        if not skip_fuse_bn:
            fuse_bn(module)

        check_fake_quantization_condition(module)

    size_in_gigabytes = sum(p.numel() * p.element_size() for p in module.parameters()) / (1 << 30)
    if size_in_gigabytes >= 2 and not export_in_tempdir:
        log.warning(
            f"Model has total parameter size larger than 2 GB ({size_in_gigabytes:.2f} GB). "
            "The ONNX must be exported with external data due to the 2 GB size limit of protobuf. "
            "(See https://github.com/onnx/onnx/issues/3275 for more details.)"
        )  # UX
        export_in_tempdir = True

    export_function = _export_path if export_in_tempdir else _export

    if skip_fuse_bn:
        if skipped_optimizers is None:
            skipped_optimizers = []
        skipped_optimizers.append("ConvBNFusion")

    if input_names is None:
        input_names = get_default_input_names(module, args)

    log.debug(f"skipped optimizers: {skipped_optimizers}")

    with FilterWarningsCausedByPasses():
        onnx_proto = export_function(
            module,
            args=args,
            export_params=export_params,
            verbose=verbose,
            training=training,
            input_names=input_names,
            output_names=output_names,
            operator_export_type=operator_export_type,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            custom_opsets=custom_opsets,
            export_modules_as_functions=export_modules_as_functions,
            skip_optimization=skip_optimization,
            skipped_optimizers=skipped_optimizers,
            check_n=check_n,
        )

    if dynamic_axis_options is not None:
        onnx_proto = dynamize(onnx_proto, dynamic_axis_options)

    onnx_proto.producer_name = f"owlite {OWLITE_VERSION} + {onnx_proto.producer_name}"
    onnx_proto.doc_string = "Processed by OwLite"

    model_dir = os.path.dirname(f)
    name, _ = os.path.splitext(os.path.basename(f))
    location = f"{name}.bin"
    abs_location = os.path.join(model_dir, location)

    log.info(f"Saving exported ONNX proto at {f} with external data {location}")  # UX
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    if abs_location is not None and os.path.isfile(abs_location):
        log.warning(f"External data file at {abs_location} will be overwritten.")  # UX
        # os.remove is required since export_to_onnx_with_external_data opens the external data file with mode='r+b'
        os.remove(abs_location)

    export_with_external_data(
        onnx_proto,
        f,
        location=location,
        ops_to_save_parameter_internally=ops_to_save_parameter_internally,
    )


# pylint: disable=missing-function-docstring, broad-exception-caught, too-many-arguments, too-many-positional-arguments
def _export(
    module: torch.nn.Module,
    args: tuple[Any, ...] | torch.Tensor,
    export_params: bool = True,
    verbose: bool = False,
    training: torch._C._onnx.TrainingMode = torch._C._onnx.TrainingMode.EVAL,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    operator_export_type: torch._C._onnx.OperatorExportTypes = torch._C._onnx.OperatorExportTypes.ONNX,
    opset_version: int | None = None,
    do_constant_folding: bool = True,
    keep_initializers_as_inputs: bool | None = None,
    custom_opsets: Mapping[str, int] | None = None,
    export_modules_as_functions: bool | Collection[type[torch.nn.Module]] = False,
    skip_optimization: bool = False,
    skipped_optimizers: list[str] | None = None,
    check_n: int = 1,
) -> ModelProto:
    with io.BytesIO() as f:
        log.debug("Running torch.onnx.export")
        torch.onnx.export(
            module,
            args=args,
            f=f,
            export_params=export_params,
            verbose=verbose,
            training=training,
            operator_export_type=operator_export_type,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            custom_opsets=custom_opsets,
            export_modules_as_functions=export_modules_as_functions,
        )
        f.seek(0)
        model_proto = _run_shape_inference(onnx.load(f))
        if skip_optimization:
            return model_proto

        optimized_proto = optimize(
            model_proto,
            input_names=input_names,
            output_names=output_names,
            skipped_optimizers=skipped_optimizers,
        )
        if optimized_proto is not model_proto:
            _check_functionality(
                optimized_proto,
                model_proto,
                check_n=check_n,
            )
        return optimized_proto


# pylint: disable-next=too-many-arguments, too-many-locals, too-many-positional-arguments
def _export_path(
    module: torch.nn.Module,
    args: tuple[Any, ...] | torch.Tensor,
    export_params: bool = True,
    verbose: bool = False,
    training: torch._C._onnx.TrainingMode = torch._C._onnx.TrainingMode.EVAL,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    operator_export_type: torch._C._onnx.OperatorExportTypes = torch._C._onnx.OperatorExportTypes.ONNX,
    opset_version: int | None = None,
    do_constant_folding: bool = True,
    keep_initializers_as_inputs: bool | None = None,
    custom_opsets: Mapping[str, int] | None = None,
    export_modules_as_functions: bool | Collection[type[torch.nn.Module]] = False,
    skip_optimization: bool = False,
    skipped_optimizers: list[str] | None = None,
    check_n: int = 1,
) -> ModelProto:
    with tempfile.TemporaryDirectory() as tempdir:
        log.debug(f"Running torch.onnx.export in {tempdir}")
        input_path = os.path.join(tempdir, "model.onnx")
        torch.onnx.export(
            module,
            args=args,
            f=input_path,
            export_params=export_params,
            verbose=verbose,
            training=training,
            operator_export_type=operator_export_type,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            custom_opsets=custom_opsets,
            export_modules_as_functions=export_modules_as_functions,
        )
        typed_onnx_path = _run_shape_inference_path(input_path, os.path.join(tempdir, "typed_model.onnx"))
        if skip_optimization:
            log.debug(f"Loading ONNX proto from {typed_onnx_path}")
            return onnx.load(typed_onnx_path)

        log.debug(f"Running ONNX optimization in {tempdir}")
        optimized_onnx_path = optimize_path(
            typed_onnx_path,
            os.path.join(tempdir, "optimized_model.onnx"),
            # Setting `size_threshold` to lower value leads to loading failure from
            # `onnxruntime.InferenceSession`, which is used for functionality checking
            # before and after the optimization. Note that this bug is only observed
            # with models larger than 2 GB.
            size_threshold=1024,
            input_names=input_names,
            output_names=output_names,
            skipped_optimizers=skipped_optimizers,
        )
        if optimized_onnx_path != typed_onnx_path:
            _check_functionality(
                optimized_onnx_path,
                input_path,
                check_n=check_n,
            )
        log.debug(f"Loading ONNX proto from {optimized_onnx_path}")
        return onnx.load(optimized_onnx_path)


def name_anonymous_nodes(onnx_proto: ModelProto) -> ModelProto:
    counts: dict[str, int] = {}
    for node in onnx_proto.graph.node:
        if node.name:
            continue

        count = counts.get(node.op_type, 0)
        postfix = f"_{count}" if count > 0 else ""
        name = f"/Anonymous/{node.op_type}{postfix}"
        log.debug(f"Naming anonymous node {nodestr(node)} -> {name}")
        node.name = name
        if node.op_type in counts:
            counts[node.op_type] += 1
        else:
            counts[node.op_type] = 1
    return onnx_proto


def get_default_input_names(module: torch.nn.Module, onnx_export_args: tuple[Any, ...] | torch.Tensor) -> list[str]:
    """Generate the default value for the `input_names` parameter of `owlite.onnx.export`.

    Args:
        module (torch.nn.Module): the module to be passed to `olt.onnx.export`
        onnx_export_args (tuple[Any, ...]): the `args` to be passed to `olt.onnx.export`. (See the `args` parameter in
        https://pytorch.org/docs/stable/onnx_torchscript.html#torch.onnx.export)

    Returns:
        list[str]: the list of input names in string.
    """
    if isinstance(module, GraphModule) and (input_signature := module.meta.get("input_signature", None)):
        return list(input_signature.keys())

    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    if isinstance(onnx_export_args, torch.Tensor):
        args, kwargs = (onnx_export_args,), {}
    else:
        if not isinstance(onnx_export_args, tuple):
            raise TypeError(f"Expected either torch.Tensor or tuple, but {onnx_export_args} is given")
        if len(onnx_export_args) > 0 and isinstance(onnx_export_args[-1], dict):
            args, kwargs = onnx_export_args[:-1], onnx_export_args[-1]
        else:
            args, kwargs = onnx_export_args, {}
    input_signature = Signature.from_module(module, args, kwargs)
    return list(input_signature.keys())


def check_fake_quantization_condition(model: GraphModule) -> bool:
    """Check if the fake quantization condition is valid.

    For TRT execution, all step sizes of fake quantizers must be positive.
    And in symmetric quantization, zero_point must be zero.
    Raise error if any of the above conditions are violated.

    Args:
        model: The model to be checked.

    Raises:
        ValueError: If any step_size of the quantizer in the model contains negative numbers,
            but the quantizer does not use clq.
        ValueError: If any zero_point of the symmetric quantizer in the model is not 0.

    Returns:
        True if all conditions are satisfied.
    """
    for name, module in model.named_modules(remove_duplicate=True):
        if isinstance(module, FakeQuantizer) and module.is_enabled:
            # check positive step_size
            if hasattr(module, "step_size") and module.qat_function is not clq_function and module.step_size.min() < 0:
                log.error(
                    f"({name}) : The step size contains negative numbers, but not using clq.\n"
                    f"{module}\nstep_size:{module.step_size.data}"
                )  # UX
                raise ValueError("The step size contains negative numbers.")
            # check symmetry
            if module.symmetric and module.zero_point.abs().max() > 0:
                raise ValueError(f"({name}) : The zero point of symmetric quantization is not 0.")

            if module.zero_point.min() < module.quant_min or module.zero_point.max() > module.quant_max:
                log.error(
                    f"({name}) : The zeropoint should be within the range ({module.quant_min, module.quant_max}).\n"
                    f"{module}\nzero_point:{module.zero_point.data}"
                )
                raise ValueError("The zero point is out of range")
    return True


def _check_functionality(
    optimized_model: bytes | str | onnx.ModelProto,
    original_model: bytes | str | onnx.ModelProto,
    *,
    check_n: int = 1,
) -> None:
    log.debug("Checking optimized model's functionality")
    try:
        if compare(
            original_model,
            optimized_model,
            n_times=check_n,
            rtol=RTOL_FP16,
            atol=ATOL_FP16,
            first_name="the ONNX model proto produced by `torch.onnx.export`",
            second_name="the ONNX model proto further optimized by OwLite",
        ):
            return
    except Exception as e:
        if STRICT_ONNX_FUNCTIONALITY_CHECKING:
            log.error(
                f"Functionality checking failed with the following exception: {e}\n"
                "You can ignore this error by setting the environment variable "
                "OWLITE_STRICT_ONNX_FUNCTIONALITY_CHECKING=0"
            )  # UX
            raise RuntimeError("ONNX functionality checking failed") from e
        log.warning(f"Failed to check the functionality of the optimized ONNX. ({e})")  # UX
        return

    if STRICT_ONNX_FUNCTIONALITY_CHECKING:
        log.error(
            "The ONNX optimization has corrupted the model's functionality. "
            "You can ignore this error by setting the environment variable OWLITE_STRICT_ONNX_FUNCTIONALITY_CHECKING=0"
        )  # UX
        raise RuntimeError("ONNX functionality corrupted")  # UX
    log.warning("The ONNX optimization has corrupted the model's functionality")  # UX


def _run_shape_inference(model: ModelProto) -> ModelProto:
    try:
        log.debug("Running ONNX shape inference")
        return infer_shapes(model, check_type=True, strict_mode=STRICT_ONNX_SHAPE_INFERENCE)
    except Exception as e:
        _handle_exception_from_shape_inference(e)
        return model


def _run_shape_inference_path(input_path: str, output_path: str) -> str:
    try:
        log.debug("Running ONNX shape inference with paths")
        infer_shapes_path(input_path, output_path, check_type=True, strict_mode=STRICT_ONNX_SHAPE_INFERENCE)
        return output_path
    except Exception as e:
        _handle_exception_from_shape_inference(e)
        return input_path


def _handle_exception_from_shape_inference(e: Exception) -> None:
    if STRICT_ONNX_SHAPE_INFERENCE:
        log.error(
            f"ONNX shape inference failed with the following exception: {e}\n"
            "You can ignore this error by setting the environment variable OWLITE_STRICT_ONNX_SHAPE_INFERENCE=0"
        )  # UX
        raise RuntimeError("ONNX shape inference failed.") from e
    log.warning(f"ONNX shape inference failed. ({e})")  # UX
