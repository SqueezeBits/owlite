"""An alternative for torch.onnx.export with extra optimizations"""
# pylint: disable=protected-access
import io
import os
import tempfile
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Optional, Union

import onnx
import onnxsim
import onnxsim.model_checking
import torch
import torch.onnx.errors
from onnx import ModelProto
from onnx.shape_inference import infer_shapes, infer_shapes_path

# pylint: disable=no-name-in-module
from onnxsim.onnxsim_cpp2py_export import simplify_path
from torch.fx.graph_module import GraphModule

from ...enums import OwLiteStatus
from ...nn import FakeQuantizer
from ...nn.functions import clq_function
from ...options import DynamicAxisOptions
from ...owlite_core.logger import log
from ..fx.transforms import clip_narrow_range_weights, fold_zp_to_bias, fuse_bn
from ..signature import Signature
from ..utils import (
    get_most_common_device,
    get_most_common_floating_point_type,
    move_tensors_to,
    nodestr,
)
from .dynamize import dynamize
from .export_with_external_data import export_with_external_data
from .model_checking import compare  # type: ignore
from .transforms import apply_onnx_transforms

# Large models (e.g. SwinTransformer) requires
# more than 50 (default) onnxsim iterations
os.environ["ONNXSIM_FIXED_POINT_ITERS"] = "100"


# pylint: disable-next=too-many-arguments, too-many-locals, invalid-name
def export(
    module: torch.nn.Module,
    args: Union[tuple[Any, ...], torch.Tensor],
    f: str,
    export_params: bool = True,
    verbose: bool = False,
    training: torch._C._onnx.TrainingMode = torch._C._onnx.TrainingMode.EVAL,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    operator_export_type: torch._C._onnx.OperatorExportTypes = torch._C._onnx.OperatorExportTypes.ONNX,
    opset_version: int = 17,
    do_constant_folding: bool = True,
    keep_initializers_as_inputs: Optional[bool] = None,
    custom_opsets: Optional[Mapping[str, int]] = None,
    export_modules_as_functions: Union[bool, Collection[type[torch.nn.Module]]] = False,
    use_fast_export: bool = True,
    apply_transforms: bool = True,
    simplify: bool = True,
    check_n: int = 1,
    skip_fuse_bn: bool = False,
    skipped_optimizers: Optional[list[str]] = None,
    dynamic_axis_options: Optional[Union[DynamicAxisOptions, dict[str, dict[str, int]]]] = None,
) -> None:
    r"""Exports a model into ONNX format.

    Args:
        module (torch.nn.Module): The model to be exported.
        args (Union[tuple[Any, ...], torch.Tensor]): Argument of a `module`.

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
        input_names (Optional[Sequence[str]], optional): Names to assign to the input nodes of the graph, in order.
            Names of `module.forward` arguments will be used when None is given. Defaults to None.
        output_names (Optional[Sequence[str]], optional): Names to assign to the output nodes of the graph, in order.
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
        keep_initializers_as_inputs (Optional[bool], optional): If True, all the initializers
            (typically corresponding to parameters) in the exported graph will also be added
            as inputs to the graph. If False, then initializers are not added as inputs to the
            graph, and only the non-parameter inputs are added as inputs. This may allow for
            better optimizations (e.g. constant folding) by backends/runtimes. Defaults to None.
        custom_opsets (Optional[Mapping[str, int]], optional): A dict with schema:

            * KEY (str): opset domain name
            * VALUE (int): opset version

            If a custom opset is referenced by ``model`` but not mentioned in this dictionary,
            the opset version is set to 1. Only custom opset domain name and version should be
            indicated through this argument. Defaults to None.
        export_modules_as_functions (Union[bool, Collection[type[torch.nn.Module]]], optional): Flag to enable
            exporting all ``nn.Module`` forward calls as local functions in ONNX. Or a set to indicate the
            particular types of modules to export as local functions in ONNX.
            This feature requires ``opset_version`` >= 15, otherwise the export will fail. This is because
            ``opset_version`` < 15 implies IR version < 8, which means no local function support.
            Module variables will be exported as function attributes. There are two categories of function
            attributes. Defaults to False.
        use_fast_export (bool, optional): If True, export process will be done in memory. If `module` with total
            parameter size larger than 2GB, this flag will be automatically set to `False`. If False, temporary
            export process will be done using temporary files. Defaults to True.
        apply_transforms (bool, optional): If True, ONNX transforms defined by SqueezeBits.inc will be applied for
            model optimization. If False, ONNX transformations will be skipped. However, turning this flag to `False`
            is experimental and might yield unexpected behavior. Defaults to True.
        simplify (bool, optional): If True, onnx-simplifier will be run. If False, onnx-simplifier will be skipped.
            Defaults to True.
        check_n (int, optional): Only available when `simplify=True`. The number of times to run check for the
            simplified ONNX proto after onnx-simplifier. Defaults to 1.
        skip_fuse_bn (bool, optional): Only available when `simplify=True`. Whether to skip batchnorm-fusion.
            Defaults to False.
        skipped_optimizers (Optional[list[str]], optional): Only available when `simplify=True`. The list of
            onnx-simplifier passes to skip. Defaults to None.
            See https://github.com/onnx/optimizer/tree/master/onnxoptimizer/passes for available passes.
        dynamic_axis_options (Optional[Union[DynamicAxisOptions, dict[str, dict[str, int]]]], optional):
            A `DynamicAxisOptions` object indicating which input tensor(s) should be configured with a dynamic axis.
            Defaults to None.

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

    if isinstance(module, GraphModule):
        if module.meta["owlite_status"] == OwLiteStatus.COMPRESSED:
            log.warning(
                "This module has not yet been calibrated. "
                "The onnx that comes out of this module may have unexpected results in accuracy and latency."
            )

        clip_narrow_range_weights(module)
        # Batch Norm Fusing
        fuse_bn(module)

        # zero point folding
        fold_zp_to_bias(module)

        check_fake_quantization_condition(module)

    device = get_most_common_device(module)
    dtype = get_most_common_floating_point_type(module)
    args = move_tensors_to(args, device, dtype)

    size_in_gigabytes = sum(p.numel() * p.element_size() for p in module.parameters()) / (1 << 30)

    if size_in_gigabytes >= 2:
        log.warning(
            f"Model has total parameter size larger than 2 GB ({size_in_gigabytes:.2f} GB)."
            '"use_fast_export" will be set to False'
        )
        use_fast_export = False

    export_function, optimize_function = (_export, _optimize) if use_fast_export else (_export_path, _optimize_path)

    if opset_version is None:
        opset_version = 17

    if input_names is None:
        input_names = get_default_input_names(module, args)
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
    )

    if skipped_optimizers is None:
        skipped_optimizers = ["fuse_qkv"]

    onnx_proto = optimize_function(
        onnx_proto,
        apply_transforms=apply_transforms,
        simplify=simplify,
        check_n=check_n,
        skip_fuse_bn=skip_fuse_bn,
        skipped_optimizers=skipped_optimizers,
    )

    if dynamic_axis_options is not None:
        if isinstance(dynamic_axis_options, dict):
            dynamic_axis_options = DynamicAxisOptions(dynamic_axis_options)
        onnx_proto = dynamize(onnx_proto, dynamic_axis_options)

    onnx_proto.producer_name = f"owlite + {onnx_proto.producer_name}"
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

    export_with_external_data(onnx_proto, f, location=location, size_threshold=0)


# pylint: disable=missing-function-docstring, broad-exception-caught
def _export(
    module: torch.nn.Module,
    args: Union[tuple[Any, ...], torch.Tensor],
    export_params: bool = True,
    verbose: bool = False,
    training: torch._C._onnx.TrainingMode = torch._C._onnx.TrainingMode.EVAL,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    operator_export_type: torch._C._onnx.OperatorExportTypes = torch._C._onnx.OperatorExportTypes.ONNX,
    opset_version: Optional[int] = None,
    do_constant_folding: bool = True,
    keep_initializers_as_inputs: Optional[bool] = None,
    custom_opsets: Optional[Mapping[str, int]] = None,
    export_modules_as_functions: Union[bool, Collection[type[torch.nn.Module]]] = False,
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
            input_names=input_names,
            output_names=output_names,
            operator_export_type=operator_export_type,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            custom_opsets=custom_opsets,
            export_modules_as_functions=export_modules_as_functions,
        )
        f.seek(0)
        onnx_proto = onnx.load(f)
        return infer_shapes(onnx_proto, check_type=True, data_prop=True)


def _export_path(
    module: torch.nn.Module,
    args: Union[tuple[Any, ...], torch.Tensor],
    export_params: bool = True,
    verbose: bool = False,
    training: torch._C._onnx.TrainingMode = torch._C._onnx.TrainingMode.EVAL,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    operator_export_type: torch._C._onnx.OperatorExportTypes = torch._C._onnx.OperatorExportTypes.ONNX,
    opset_version: Optional[int] = None,
    do_constant_folding: bool = True,
    keep_initializers_as_inputs: Optional[bool] = None,
    custom_opsets: Optional[Mapping[str, int]] = None,
    export_modules_as_functions: Union[bool, Collection[type[torch.nn.Module]]] = False,
) -> ModelProto:
    with tempfile.TemporaryDirectory() as tempdir:
        model_path = os.path.join(tempdir, "model.onnx")
        log.debug(f"Running torch.onnx.export with output path: {model_path}")
        torch.onnx.export(
            module,
            args=args,
            f=model_path,
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
        )
        output_path = os.path.join(tempdir, "typed_model.onnx")
        infer_shapes_path(model_path, output_path, check_type=True, data_prop=True)
        log.debug(f"Loading ONNX proto from {output_path}")
        return onnx.load(output_path)


def _optimize(
    onnx_proto: ModelProto,
    apply_transforms: bool = True,
    simplify: bool = True,
    check_n: int = 1,
    skip_fuse_bn: bool = False,
    skipped_optimizers: Optional[list[str]] = None,
) -> ModelProto:
    modified_proto = onnx_proto
    try:
        if simplify:
            log.debug("Running onnxsim.simplify")
            modified_proto, _ = onnxsim.simplify(
                modified_proto,
                check_n=0,
                skip_fuse_bn=skip_fuse_bn,
                skipped_optimizers=skipped_optimizers,
            )

            # onnxsim produces anonymous nodes problematic for creating ONNXToFXMap
            modified_proto = name_anonymous_nodes(modified_proto)
            if len([*modified_proto.graph.node]) == 0:
                log.warning("All nodes are constant-folded by onnxsim.")
    except Exception as e:
        log.warning(f"ONNX simplifier failed with error: {e}")

    if apply_transforms:
        modified_proto = apply_onnx_transforms(modified_proto)

    if modified_proto is not onnx_proto:
        log.debug("Checking modified model")
        ok = compare(modified_proto, onnx_proto, n_times=check_n)
        if not ok:
            log.warning("The output has been changed after the optimization")

    return modified_proto


def _optimize_path(
    onnx_proto: ModelProto,
    apply_transforms: bool = True,
    simplify: bool = True,
    check_n: int = 1,
    skip_fuse_bn: bool = False,
    skipped_optimizers: Optional[list[str]] = None,
) -> ModelProto:
    with tempfile.TemporaryDirectory() as tempdir:
        output_path = os.path.join(tempdir, "model.onnx")
        onnx.save(
            onnx_proto,
            output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.bin",
        )
        modified_proto = onnx_proto
        try:
            if simplify:
                log.debug("Running onnxsim.simplify_path")
                simplified_output_path = os.path.join(tempdir, "simple_model.onnx")
                if skip_fuse_bn:
                    if skipped_optimizers is None:
                        skipped_optimizers = []
                    skipped_optimizers.append("fuse_bn_into_conv")
                ok = simplify_path(
                    output_path,
                    simplified_output_path,
                    skipped_optimizers,
                    True,
                    True,
                    10 * (1 << 30),  # 10 GB
                )
                if not ok:
                    log.warning("ONNX simplifier failed")
                    return onnx_proto
                log.debug(f"Loading simplified ONNX proto from {simplified_output_path}")
                modified_proto = onnx.load(simplified_output_path)
                # onnxsim produces anonymous nodes problematic for creating ONNXToFXMap
                modified_proto = name_anonymous_nodes(modified_proto)
                if len([*modified_proto.graph.node]) == 0:
                    log.warning("All nodes are constant-folded by onnxsim.")
        except Exception as e:
            log.warning(f"ONNX simplifier failed with error: {e}")

        if apply_transforms:
            transformed_output_path = os.path.join(tempdir, "transformed_model.onnx")
            log.debug(f"Applying ONNX transforms with output path: {transformed_output_path}")
            modified_proto = apply_onnx_transforms(modified_proto, transformed_output_path)

        if modified_proto is not onnx_proto:
            log.debug("Checking modified model")
            ok = compare(simplified_output_path, output_path, check_n, None, None, None)
            if not ok:
                log.warning("The output has been changed after the optimization")
        return modified_proto


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


def get_default_input_names(
    module: torch.nn.Module, onnx_export_args: Union[tuple[Any, ...], torch.Tensor]
) -> list[str]:
    """Decides the default value of `input_names` parameter for `olt.onnx.export` based on the signature of the module's
    forward method and the onnx_export_args.

    Args:
        module (torch.nn.Module): the module to be passed to `olt.onnx.export`
        onnx_export_args (tuple[Any, ...]): the `args` to be passed to `olt.onnx.export`. (See the `args` parameter in
        https://pytorch.org/docs/stable/onnx_torchscript.html#torch.onnx.export)

    Returns:
        list[str]: the list of input names in string.
    """
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
    input_shape_signature = Signature.from_module(module, args, kwargs)
    return [*(pair[0] for pair in input_shape_signature)]


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
            if module.symmetric and module.zero_point.amax() > 0:
                raise ValueError(f"({name}) : The zero point of symmetric quantization is not 0.")
    return True
