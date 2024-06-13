# pylint: disable=protected-access, too-many-statements
import inspect
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch._dynamo as torch_dynamo
import torch.utils._pytree as pytree
from torch import _guards
from torch._dispatch.python import enable_python_dispatcher
from torch.fx.graph_module import GraphModule
from torch.nn.parallel import DataParallel, DistributedDataParallel

from ...enums import ModelStatus
from ...owlite_core.logger import log
from ..config import FORCE_GRAPH_MODULE_COMPATIBILITY
from ..signature import Signature
from ..utils import (
    get_most_common_device,
    get_most_common_floating_point_type,
    move_tensors_to,
)
from .transforms import apply_graph_module_transforms


# pylint: disable-next=too-many-locals
def symbolic_trace(model: torch.nn.Module, *args: Any, **kwargs: Any) -> GraphModule:
    """Symbolically trace the input `model` to convert it into a GraphModule.

    In order for the tracing to be successful, the `model` must be able to pass `torch.compile(model, fullgraph=True)`.

    Args:
        model (torch.nn.Module): a torch.nn.Module instance.
        *args: the example input(s) that would be passed to the model's forward method.
        **kwargs: the example input(s) that would be passed to the model's forward method.

    Raises:
        TypeError: if the `model` is not an instance of `torch.nn.Module`
        RuntimeError: if the tracing fails.

    Returns:
        GraphModule: the converted GraphModule.
    """
    given_type = type(model)
    if isinstance(model, DataParallel | DistributedDataParallel):
        log.error(
            f"{given_type} is not supported by symbolic trace, please use 'attribute' module to unwrap model "
            f"from {given_type}. Try owlite.fx.symbolic_trace(model.module, ...)"
        )

    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected torch.nn.Module instance but object of type {given_type} given: {model}")

    training_status = model.training
    # move input args and kwargs to model device
    device = get_most_common_device(model)
    dtype = get_most_common_floating_point_type(model)
    log.debug(f"Tracing with device={device}, dtype={dtype}")

    args = move_tensors_to(args, device, dtype)
    kwargs = move_tensors_to(kwargs, device, dtype)

    original_signature = inspect.signature(model.forward)
    flat_args, in_spec = pytree.tree_flatten((args, kwargs))
    graph_module: GraphModule | None = None
    graph_captured_input = None
    graph_captured_result: tuple[torch.Tensor, ...] | None = None
    example_inputs: Sequence[Any] | None = None
    fake_mode = None

    def dynamo_normalization_capturing_compiler(
        gm: GraphModule, inner_example_inputs: Sequence[Any]
    ) -> Callable[..., Any]:
        nonlocal graph_module
        assert (
            graph_module is None
        ), "Tried to emit a second graph during export. Tracing through 'f' must produce a single graph."
        graph_module = gm

        nonlocal fake_mode, example_inputs
        # NB: do NOT pass inner_example_inputs here, we are detecting the
        # Dynamo allocated fake mode, which should be DISTINCT from a
        # potential outer ambient fake mode which the user provided.
        # example_inputs is always the user specified inputs, so they
        # would have the wrong fake mode attached to them
        fake_mode = _guards.detect_fake_mode()
        example_inputs = inner_example_inputs

        def result_capturing_wrapper(*graph_inputs: Any) -> Any:
            nonlocal graph_captured_result
            nonlocal graph_captured_input

            graph_captured_input = graph_inputs
            assert graph_module is not None

            named_parameters = dict(graph_module.named_parameters(remove_duplicate=False))
            named_buffers = dict(graph_module.named_buffers(remove_duplicate=False))

            ambient_fake_mode = (
                _guards.detect_fake_mode(graph_inputs)
                if _guards.detect_fake_mode(graph_inputs) is not None
                else fake_mode
            )

            with ambient_fake_mode, enable_python_dispatcher():
                params_and_buffers = {
                    **dict(named_parameters),
                    **dict(named_buffers),
                }
                fake_params_buffers = {}

                for name, value in params_and_buffers.items():
                    fake_params_buffers[name] = ambient_fake_mode.from_tensor(value, static_shapes=True)

                fake_graph_inputs = pytree.tree_map(ambient_fake_mode.from_tensor, graph_inputs)
                graph_captured_result = torch.func.functional_call(graph_module, fake_params_buffers, fake_graph_inputs)

            return graph_captured_result

        return result_capturing_wrapper

    torch_dynamo.eval_frame.remove_from_cache(model)
    optimized_model = torch_dynamo.optimize_assert(backend=dynamo_normalization_capturing_compiler, export=True)(model)
    output = optimized_model(*args, **kwargs)
    torch_dynamo.eval_frame.remove_from_cache(model)

    assert graph_module is not None, "Failed to create torch.fx.GraphModule while running optimized model"
    assert example_inputs is not None

    if FORCE_GRAPH_MODULE_COMPATIBILITY:
        torch_dynamo.eval_frame.check_signature_rewritable(graph_module)

        # pylint: disable-next=not-an-iterable
        example_fake_inputs = [fake_mode.from_tensor(t) for t in example_inputs]  # type: ignore[union-attr]
        rewrite_signature_args = [
            original_signature,
            graph_module,
            fake_mode,
            flat_args,
            in_spec,
            example_fake_inputs,
            graph_captured_input,
            graph_captured_result,
            output,
        ]
        if torch.__version__ >= (2, 1, 3):  # type: ignore[operator]
            flat_args_dynamic_dims: list[dict[str, Any]] = [{} for _ in flat_args]
            rewrite_signature_args.append(flat_args_dynamic_dims)
        # pylint: disable-next=no-value-for-parameter
        graph_module = torch_dynamo.eval_frame.rewrite_signature(*rewrite_signature_args)

    graph_module = apply_graph_module_transforms(graph_module)  # type: ignore[arg-type]
    graph_module.train(training_status)
    graph_module.meta["status"] = ModelStatus.TRACED
    graph_module_input_signature = Signature.from_module(graph_module, args, kwargs)
    graph_module_input_signature.warn_signature_change(dict(original_signature.parameters.items()))
    graph_module.meta["input_signature"] = graph_module_input_signature

    return graph_module
