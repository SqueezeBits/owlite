import json
from typing import Optional

import requests

from ...backend.utils import extract_input_signature_from_onnx_proto
from ...utils.file_utils import upload_file_to_url
from ..api_base import MAIN_API_BASE


def create_run(project_id: str, baseline_name: str, run_name: str) -> None:
    """Creates an experiment.

    Args:
        project_id (str): The id of a project.
        baseline_name (str): The name of a baseline.
        run_name (str): The name of a new experiment.

    Raises:
        HTTPError: When request was not successful.
    """

    payload = {
        "project_id": project_id,
        "baseline_name": baseline_name,
        "run_name": run_name,
    }

    res = MAIN_API_BASE.post("/projects/runs", json=payload)
    assert isinstance(res, dict)


def get_run_info(project_id: str, baseline_name: str, run_name: str) -> Optional[dict]:
    """Gets information of an experiment.

    Args:
        project_id (str): The id of a project.
        baseline_name (str): The name of a baseline.
        run_name (str): The name of an experiment.

    Returns:
        Optional[dict]: The information of an experiment if exists, None otherwise.

    Raises:
        HTTPError: When request was not successful.
    """

    payload = {
        "project_id": project_id,
        "baseline_name": baseline_name,
        "run_name": run_name,
    }

    try:
        res = MAIN_API_BASE.post("/projects/runs/info", json=payload)

    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return None

        raise e

    assert isinstance(res, dict)
    return res


def copy_run(project_id: str, baseline_name: str, duplicate_from: str, run_name: str) -> str:
    """Copies existing experiment and create a new experiment. Compression configuration is also cloned.

    Args:
        project_id (str): The id of a project.
        baseline_name (str): The name of a baseline.
        duplicate_from (str): The name of an experiment to clone.
        run_name (str): The name of a new experiment.

    Returns:
        str: The name of a created experiment.

    Raises:
        HTTPError: When request was not successful.
    """

    payload = {
        "project_id": project_id,
        "baseline_name": baseline_name,
        "run_name": duplicate_from,
        "new_run_name": run_name,
    }

    resp = MAIN_API_BASE.post("/projects/runs/copy", json=payload)
    assert isinstance(resp, dict)
    return str(resp["name"])


def upload_run_onnx_proto(
    project_id: str,
    baseline_name: str,
    run_name: str,
    onnx_path: str,
    dynamic_axes: Optional[dict[str, dict[int, dict[str, int]]]] = None,
) -> None:
    """Uploads experiment's onnx proto and graph module. Note that parameters are not uploaded.

    Args:
        project_id (str): The id of a project.
        baseline_name (str): The name of a baseline.
        run_name (str): The name of an experiment.
        onnx_path (str): The path to experiment onnx proto file.
        dynamic_axes (Optional[dict[str, dict[int, dict[str, int]]]], optional): Dynamic axes setting,
            please refer to owlite.onnx.export for detail.

    Raises:
        HTTPError: When request was not successful.
    """

    input_signature = extract_input_signature_from_onnx_proto(onnx_path)
    if dynamic_axes is not None:
        new_input_signature = []
        for name, shape in input_signature:
            axis_setting = dynamic_axes.get(name)
            if axis_setting is not None:
                axis = next(iter(axis_setting))
                setting = axis_setting.get(axis)
                assert setting is not None
                range_setting = [
                    setting.get("min"),
                    setting.get("opt"),
                    setting.get("max"),
                    setting.get("test"),
                ]
                shape[axis] = range_setting  # type: ignore
            new_input_signature.append((name, shape))
        input_signature = new_input_signature

    payload = {
        "project_id": project_id,
        "baseline_name": baseline_name,
        "run_name": run_name,
        "input_shape": json.dumps(input_signature),
    }

    file_dest_url = MAIN_API_BASE.post("/projects/runs/data/upload", json=payload)

    assert file_dest_url is not None and isinstance(file_dest_url, str)
    file_upload_resp = upload_file_to_url(onnx_path, file_dest_url)

    if not file_upload_resp.ok:
        file_upload_resp.raise_for_status()


def get_benchmark_key(project_id: str, baseline_name: str, run_name: str) -> str:
    """Gets a key to identify a benchmark job.

    Args:
        project_id (str): The id of a project.
        baseline_name (str): The name of a baseline.
        run_name (str): The name of an experiment.

    Returns:
        str: A key to identify a benchmark job.

    Raises:
        HTTPError: When request was not successful.
    """

    payload = {
        "project_id": project_id,
        "baseline_name": baseline_name,
        "run_name": run_name,
    }

    resp = MAIN_API_BASE.post("/projects/runs/keys", json=payload)

    assert isinstance(resp, str)
    return resp


def update_run_info(
    project_id: str,
    baseline_name: str,
    run_name: str,
    logs: str,
) -> None:
    """Updates information for a specific experiment with model metrics.

    Args:
        project_id (str): The id of a project.
        baseline_name (str): The name of a baseline.
        run_name (str): The name of an experiment.
        logs (str): Logs to be stored in the database.

    Raises:
        HTTPError: When request was not successful.
    """

    payload = {
        "project_id": project_id,
        "baseline_name": baseline_name,
        "run_name": run_name,
        "logs": logs,
    }

    resp = MAIN_API_BASE.post("/projects/runs/update", json=payload)
    assert isinstance(resp, str)
