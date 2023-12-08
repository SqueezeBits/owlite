"""API wrapper module for dove api calls"""

import base64
import json

import onnx
from torch.fx.graph_module import GraphModule
from torch.nn.parallel import DataParallel, DistributedDataParallel

from ...backend.fx.serialize import serialize
from ...backend.utils import extract_input_signature_from_onnx_proto
from ...logger import log
from ..api_base import DOVE_API_BASE


def upload_baseline(
    project_id: str,
    baseline_name: str,
    onnx_path: str,
    model: GraphModule,
) -> None:
    """Uploads baseline's onnx proto and graph module.

    Args:
        project_id (str): The id of a project.
        baseline_name (str): The name of a baseline.
        onnx_path (str): The path to baseline onnx proto file.
        model (GraphModule): The traced graph module.

    Raises:
        TypeError: When the `model` is not an instance of `torch.fx.GraphModule`.
        HTTPError: When the request was not successful.
    """
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        _model_type = f"torch.nn.parallel.{type(model).__name__}"
        log.error(
            f"{_model_type} is not supported by upload_baseline, please use 'attribute' module to unwrap model "
            f"{_model_type}. Try owlite.api.dove.doves.upload_baseline(..., model = model.module)"
        )
        raise TypeError(f"{_model_type} is not supported by upload_baseline")
    if not isinstance(model, GraphModule):
        raise TypeError(f"model of upload_baseline must be GraphModule, but got {type(model)}")

    proto = onnx.load(onnx_path, load_external_data=False)
    input_shape = json.dumps(extract_input_signature_from_onnx_proto(proto))

    payload = {
        "project_id": project_id,
        "baseline_name": baseline_name,
        "gm": serialize(model),
        "onnx": base64.b64encode(proto.SerializeToString()),
        "input_shape": input_shape,
    }

    DOVE_API_BASE.post("/upload", payload)


def get_configuration(
    project_id: str,
    baseline_name: str,
    run_name: str,
) -> str:
    """Gets configuration options to apply.

    Args:
        project_id (str): The id of a project.
        baseline_name (str): The name of a baseline.
        run_name (str): The name of a run.

    Returns:
        str: The compiled configuration string.

    Raises:
        HTTPError: When request was not successful.
    """
    payload = {
        "project_id": project_id,
        "baseline_name": baseline_name,
        "run_name": run_name,
    }
    resp = DOVE_API_BASE.post("/compile", json=payload)
    assert isinstance(resp, dict)

    return json.dumps(resp)
