from pydantic import BaseModel


class Tokens(BaseModel):
    """Represents tokens.

    Attributes:
        access_token (str): access token for OwLite login.
        refresh_token (str): refresh token for OwLite login.
    """

    access_token: str
    refresh_token: str
