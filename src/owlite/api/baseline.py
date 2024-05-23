# pylint: disable=duplicate-code
import base64
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import onnx
import requests
import torch
from packaging.version import Version
from torch.fx.graph_module import GraphModule
from typing_extensions import Self

from ..backend.fx import serialize
from ..backend.signature import Signature
from ..owlite_core.api_base import DOVE_API_BASE, MAIN_API_BASE
from ..owlite_core.cache.device import Device
from ..owlite_core.constants import OWLITE_VERSION
from ..owlite_core.logger import log
from .benchmarkable import Benchmarkable
from .project import Project

if TYPE_CHECKING:
    from .experiment import Experiment


@dataclass
class Baseline(Benchmarkable):
    """The OwLite baseline."""

    project: Project
    experiments: dict[str, "Experiment"] = field(default_factory=dict)
    input_signature: Signature | None = field(default=None)

    @property
    def home(self) -> str:
        return os.path.join(self.project.home, self.name)

    @property
    def label(self) -> str:
        return "_".join((self.project.name, self.name, self.name))

    @property
    def url(self) -> str:
        # TODO (huijong): make this url point to the insight page of the baseline.
        return self.project.url

    @classmethod
    def create(cls, project: Project, name: str, device: Device) -> Self:
        """Create a baseline named `name` under the given `project`.

        Args:
            project (Project): An OwLite project.
            name (str): The name for the baseline to be created.
            device (Device): The device for benchmarking this baseline.

        Returns:
            Baseline: The created baseline
        """
        extra_payload = {
            "framework": device.runtime.value,
            "owlite_version": str(OWLITE_VERSION),
            "torch_version": str(torch.__version__),
        }

        resp = MAIN_API_BASE.post(
            "/projects/baselines",
            json=Baseline(name=name, project=project, device=device).payload(**extra_payload),
        )
        assert isinstance(resp, dict)
        name_from_resp = resp["baseline_name"]
        if name_from_resp != name:
            log.warning(
                f"The baseline '{name}' already exists. Created a new baseline '{name_from_resp}' at {project}"
            )  # UX
        baseline = cls(name=name_from_resp, project=project, device=device)
        log.info(f"Created a new {baseline}")  # UX
        project.baseline = baseline
        return baseline

    @classmethod
    def load(cls, project: Project, name: str, device: Device) -> Self | None:
        """Load the existing baseline with name `name` from the given `project`.

        Args:
            project (Project): An OwLite project
            name (str): The name of an existing baseline in the project
            device (Device): The device for which the baseline is being loaded.

        Raises:
            requests.exceptions.HTTPError: When the server responded with an unexpected status code.
            RuntimeError: When loading a baseline created in a different environment.

        Returns:
            Baseline | None: the existing baseline if found, `None` otherwise.
        """
        try:
            resp = MAIN_API_BASE.post(
                "/projects/baselines/check",
                json=Baseline(name=name, project=project, device=device).payload(),
            )
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return None
            raise e

        assert isinstance(resp, dict)

        baseline_owlite_version = Version(resp["owlite_version"] or "0.0.0")
        baseline_torch_version = Version(resp["torch_version"] or "0.0.0")
        current_torch_version = Version(str(torch.__version__))

        if baseline_owlite_version != OWLITE_VERSION or baseline_torch_version != current_torch_version:
            detail = (
                f"\n{baseline_owlite_version}(baseline OwLite version) != {OWLITE_VERSION}(current OwLite version)"
                if baseline_owlite_version != OWLITE_VERSION
                else ""
            )
            detail += (
                f"\n{baseline_torch_version}(baseline PyTorch version) != {torch.__version__}(current PyTorch version)"
                if baseline_torch_version != current_torch_version
                else ""
            )
            log.error(
                "It seems like you are trying to load a baseline created in an incompatible environment. "
                f"Please check your current environment and try again{detail}"
            )  # UX
            raise RuntimeError("Version mismatch")

        input_signature = Signature.from_str(resp["input_shape"]) if resp["input_shape"] else None
        baseline = cls(
            name=name,
            project=project,
            input_signature=input_signature,
            device=Device(manager=device.manager, **resp),
        )

        project.baseline = baseline
        return baseline

    def upload(
        self,
        proto: onnx.ModelProto,
        model: GraphModule,
    ) -> None:
        assert self.input_signature
        log.debug(f"Baseline signature: {self.input_signature}")

        DOVE_API_BASE.post(
            "/upload",
            data=self.payload(
                gm=serialize(model),
                onnx=base64.b64encode(proto.SerializeToString()).decode("utf-8"),
                input_shape=self.input_signature.dumps(),
                **self.version_payload,
            ),
        )

        log.info("Uploaded the model excluding parameters")  # UX

    def payload(self, **kwargs: str | int) -> dict[str, str | int]:
        p: dict[str, str | int] = {
            "project_id": self.project.id,
            "baseline_name": self.name,
        }
        p.update(kwargs)
        return p

    def __str__(self) -> str:
        return f"baseline '{self.name}' in the {self.project}"
