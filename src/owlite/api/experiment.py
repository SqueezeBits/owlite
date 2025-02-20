# pylint: disable=duplicate-code
import os
from dataclasses import dataclass, field
from functools import cached_property

import onnx
import requests
from torch.fx.graph_module import GraphModule
from typing_extensions import Self

from ..backend.signature import Signature
from ..core.api_base import DOVE_API_BASE, MAIN_API_BASE
from ..core.cache.device import Device
from ..core.cache.workspace import Workspace
from ..core.constants import FX_CONFIGURATION_FORMAT_VERSION, OWLITE_VERSION
from ..core.logger import log
from ..enums import PricePlan
from ..options import CompressionOptions
from .baseline import Baseline
from .benchmarkable import Benchmarkable
from .project import Project
from .utils import upload_file_to_url


@dataclass
class Experiment(Benchmarkable):
    """The OwLite experiment."""

    baseline: Baseline
    has_config: bool
    input_signature: Signature | None = field(default=None)

    @property
    def workspace(self) -> Workspace:
        """The parent workspace for this experiment."""
        return self.baseline.project.workspace

    @property
    def plan(self) -> PricePlan:
        """The price plan for this experiment."""
        return self.workspace.plan

    @property
    def project(self) -> Project:
        """The parent project for this experiment."""
        return self.baseline.project

    @property
    def url(self) -> str:
        # TODO (huijong): make this url point to the insight page comparing the experiment against its baseline.
        return self.project.url

    @property
    def home(self) -> str:
        return os.path.join(self.baseline.home, self.name)

    @property
    def label(self) -> str:
        return "_".join((self.project.name, self.baseline.name, self.name))

    @cached_property
    def config(self) -> CompressionOptions:
        """The configuration for this experiment."""
        try:
            resp = DOVE_API_BASE.post(
                "/compile",
                json=self.payload(format_version=str(FX_CONFIGURATION_FORMAT_VERSION), **self.version_payload),
            )
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 426:
                log.error(
                    f"Your current version ({OWLITE_VERSION}) is not supported. "
                    "Please update the package to the latest version with the following command: "
                    "pip install owlite --extra-index-url https://pypi.squeezebits.com/ --upgrade "
                )  # UX
            raise e
        assert isinstance(resp, dict)
        return CompressionOptions.deserialize(resp)

    @classmethod
    def create(cls, baseline: Baseline, name: str, device: Device) -> Self:
        """Create a new experiment for the baseline.

        Args:
            baseline (Baseline): A baseline
            name (str): The name of the experiment to be created
            device (Device): The device for benchmarking this experiment.

        Returns:
            Experiment: The newly created experiment
        """
        try:
            res = MAIN_API_BASE.post(
                "/projects/runs",
                json=baseline.payload(run_name=name),
            )
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 400:
                err_msg = e.response.json()
                if "is invalid" in err_msg["detail"]:
                    log.error(
                        "Baseline model is invalid. "
                        "You can only create an experiment with an uploaded baseline model. "
                        "Please check if OwLite successfully uploaded the baseline model. "
                        "If not, try using `owl.export(model).`"
                    )  # UX
            elif e.response is not None and e.response.status_code == 403:
                log.error(
                    "You can create up to 20 experiments in a single Free Plan Workspace. In this execution, "
                    "OwLite functions will not be executed. Please delete any unused Experiment and try again."
                )  # UX
            raise e
        assert isinstance(res, dict)
        experiment = cls(name=name, baseline=baseline, device=device, has_config=False)
        log.info(f"Created a new {experiment}")  # UX
        baseline.experiments[name] = experiment
        return experiment

    @classmethod
    def load(cls, baseline: Baseline, name: str, device: Device, *, verbose: bool = True) -> Self | None:
        """Load the existing experiment named `name` for the given `baseline`.

        Args:
            baseline (Baseline): The baseline holding the experiment
            name (str): The name of the experiment to load
            device (Device): The device for benchmarking the experiment.
            verbose (bool, optional): If True, prints error message when the experiment is not found. Defaults to True.

        Raises:
            e (requests.exceptions.HTTPError): When an unexpected HTTP status code is returned.

        Returns:
            Experiment | None: The loaded experiment if it is found, `None` otherwise.
        """
        try:
            res = MAIN_API_BASE.post("/projects/runs/info", json=baseline.payload(run_name=name))
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                if verbose:
                    log.error(
                        f"No such experiment: {name}. Please check if the experiment name is correct "
                        f"or create a new one at {baseline.project.url}"
                    )  # UX
                return None
            raise e

        assert isinstance(res, dict)
        experiment = cls(name=name, baseline=baseline, device=device, has_config=bool(res.get("config_id", "")))
        log.info(f"Loaded the existing {experiment}")  # UX
        baseline.experiments[name] = experiment
        return experiment

    @classmethod
    def load_or_create(cls, baseline: Baseline, name: str, device: Device) -> Self:
        """Load the experiment named `name` for the given `baseline` if it already exists, creates a new one otherwise.

        Args:
            baseline (Baseline): The baseline holding the experiment.
            name (str): The name of the experiment to be loaded or created.
            device (Device): The device for creating or benchmarking the experiment.

        Returns:
            Experiment: The loaded or newly created experiment.
        """
        experiment = cls.load(baseline, name, device, verbose=False) or cls.create(baseline, name, device)

        if experiment.has_config:
            log.info(f"Compression configuration found for '{experiment.name}'")  # UX
        else:
            log.warning(f"No compression configuration found for '{experiment.name}'")  # UX

        return experiment

    def clone(self, name: str) -> Self:
        """Clone this experiment.

        Args:
            name (str): The name of the new experiment.

        Raises:
            e (requests.exceptions.HTTPError): When an unexpected HTTP status code is returned.
            RuntimeError: When the experiment to duplicate does not have compression configuration.

        Returns:
            Experiment: The cloned experiment.
        """
        try:
            resp = MAIN_API_BASE.post("/projects/runs/copy", json=self.payload(new_run_name=name))
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 400:
                log.error(
                    f"Cannot duplicate experiment. Experiment '{self.name}' doesn't have compression configuration"
                )  # UX
                raise RuntimeError("Compression configuration not found") from e
            if e.response is not None and e.response.status_code == 403:
                log.error(
                    "You can create up to 20 experiments in a single Free Plan Workspace. In this execution, "
                    "OwLite functions will not be executed. Please delete any unused Experiment and try again."
                )  # UX
            raise e
        assert isinstance(resp, dict)
        cloned_experiment = type(self)(
            name=resp["name"], baseline=self.baseline, device=self.device, has_config=self.has_config
        )
        log.info(
            f"Copied compression configuration from the {self} to the new experiment '{cloned_experiment.name}'"
        )  # UX
        return cloned_experiment

    def upload(
        self,
        proto: onnx.ModelProto | None = None,
        model: GraphModule | None = None,
    ) -> None:
        assert self.input_signature
        log.debug(f"Experiment signature: {self.input_signature}")

        try:
            file_dest_url = MAIN_API_BASE.post(
                "/projects/runs/data/upload",
                json=self.payload(input_shape=self.input_signature.dumps(), **self.version_payload),
            )
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 400:
                err_msg = e.response.json()
                if "shapes mismatch" in err_msg["detail"]:
                    log.error(
                        "Input signature of current experiment does not match with baseline's. "
                        f"Please compare current input signature: {self.input_signature} "
                        f"and baseline input signature: {self.baseline.input_signature}"
                    )  # UX
                    raise RuntimeError("Input signature mismatch") from e
            raise e
        assert file_dest_url is not None and isinstance(file_dest_url, str)
        upload_file_to_url(self.onnx_path, file_dest_url)

    def payload(self, **kwargs: str | int) -> dict[str, str | int]:
        p: dict[str, str | int] = {
            "workspace_id": self.workspace.id,
            "project_id": self.project.id,
            "baseline_name": self.baseline.name,
            "run_name": self.name,
        }
        p.update(kwargs)
        return p

    def __str__(self) -> str:
        return f"experiment '{self.name}' for the {self.baseline}"
