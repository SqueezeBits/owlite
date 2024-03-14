from typing import Optional

from pydantic import BaseModel, Field

from ..constants import (
    NEST_URL,
    OWLITE_DOVE_API_BASE_URL,
    OWLITE_FRONT_BASE_URL,
    OWLITE_MAIN_API_BASE_URL,
)


# pylint:disable=too-few-public-methods
class BaseURLs(BaseModel):
    """Represents base urls.

    Attributes:
        FRONT (str): The url for OwLite front server.
        MAIN (str): The url for OwLite main server.
        DOVE (str): The url for OwLite Dove server.
    """

    FRONT: str = Field(default=OWLITE_FRONT_BASE_URL)
    MAIN: str = Field(default=OWLITE_MAIN_API_BASE_URL)
    DOVE: str = Field(default=OWLITE_DOVE_API_BASE_URL)
    NEST: str = Field(default=NEST_URL)

    def set(self, name: str, url: Optional[str] = None) -> None:
        """Sets the given URL to input value or its default value.

        Args:
            name (str): The name of the URL to reset.
            url (str, None): The address to change. If None, reset it.
        """
        if name in self.model_fields:
            if url:
                setattr(self, name, url)
            else:
                setattr(self, name, self.model_fields[name].default)
        else:
            raise ValueError(f"Invalid name: {name}")
