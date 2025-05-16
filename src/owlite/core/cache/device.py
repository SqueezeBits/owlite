from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from ...enums.runtime import Runtime


class Device(BaseModel):
    """Represents a device.

    Attributes:
        name (str): The name of the device.
        runtime (Runtime): The runtime associated with the device.
        runtime_extra (str | None): Extra information about the device determined in code runtime.
    """

    model_config = ConfigDict(extra="ignore")
    name: str = Field(validation_alias=AliasChoices("name", "device_name"))
    runtime: Runtime = Field(default=Runtime.TensorRT, validation_alias=AliasChoices("framework", "runtime"))
    runtime_extra: str | None = Field(default=None, exclude=True)

    def __str__(self) -> str:
        return f"{self.runtime_extra or self.name} [{self.runtime.name}]"
