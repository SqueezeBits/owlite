# pylint: disable=duplicate-code
import base64
import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Union

import onnx
import requests
from torch.fx.graph_module import GraphModule
from typing_extensions import Self

from ..backend.fx import serialize
from ..backend.signature import DynamicSignature, Signature
from ..owlite_core.api_base import DOVE_API_BASE, MAIN_API_BASE
from ..owlite_core.logger import log
from .benchmarkable import Benchmarkable
from .project import Project

if TYPE_CHECKING:
    from .experiment import Experiment


@dataclass
class Baseline(Benchmarkable):
    """The OwLite baseline"""

    project: Project
    experiments: dict[str, "Experiment"] = field(default_factory=dict)
    input_signature: Optional[Union[Signature, DynamicSignature]] = field(default=None)

    @property
    def exists(self) -> bool:
        """Checks if the baseline represented by this object actually exists

        Raises:
            e (requests.exceptions.HTTPError): When unexpected error has been thrown.

        Returns:
            bool: True if it exists, False otherwise.
        """
        try:
            resp = MAIN_API_BASE.post(
                "/projects/baselines/check",
                json=self.payload(),
            )
            assert isinstance(resp, bool)
            return resp
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return False

            raise e

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
    def create(cls, project: Project, name: str) -> Self:
        """Creates a baseline named `name` under the given `project`

        Args:
            project (Project): An OwLite project.
            name (str): The name for the baseline to be created.

        Returns:
            Baseline: The created baseline
        """
        resp = MAIN_API_BASE.post(
            "/projects/baselines",
            json=Baseline(name=name, project=project, input_signature=None).payload(),
        )
        assert isinstance(resp, dict)
        name_from_resp = resp["baseline_name"]
        if name_from_resp != name:
            log.warning(
                f"The baseline '{name}' already exists. Created a new baseline '{name_from_resp}' at {project}"
            )  # UX
        baseline = cls(name=name_from_resp, project=project, input_signature=None)
        log.info(f"Created a new {baseline}")  # UX
        project.baseline = baseline
        return baseline

    @classmethod
    def load(cls, project: Project, name: str) -> Optional[Self]:
        """Loads the existing baseline with name `name` from the given `project`

        Args:
            project (Project): An OwLite project
            name (str): The name of an existing baseline in the project

        Returns:
            Optional[Baseline]: the existing baseline if found, `None` otherwise.
        """
        baseline = cls(name=name, project=project, input_signature=None)
        if baseline.exists:
            project.baseline = baseline
            return baseline
        return None

    def upload(
        self,
        proto: onnx.ModelProto,
        model: Optional[GraphModule],
    ) -> None:
        assert model is not None
        self.input_signature = Signature.from_onnx(proto)

        log.debug(f"Baseline signature: {self.input_signature}")
        DOVE_API_BASE.post(
            "/upload",
            data=self.payload(
                gm=serialize(model),
                onnx=base64.b64encode(proto.SerializeToString()).decode("utf-8"),
                input_shape=json.dumps(self.input_signature),
            ),
        )

        log.info("Uploaded the model excluding parameters")  # UX

    def payload(self, **kwargs: str) -> dict[str, str]:
        p = {
            "project_id": self.project.id,
            "baseline_name": self.name,
        }
        p.update(kwargs)
        return p

    def __repr__(self) -> str:
        return f"baseline '{self.name}' in the {self.project}"
