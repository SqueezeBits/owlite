from .cache import OWLITE_CACHE_PATH
from .cache.base_urls import BaseURLs
from .cache.text import read_text, write_text
from .cache.tokens import Tokens
from .cache.workspace import Workspace
from .logger import log


class OwLiteSettings:
    """Handle OwLite settings including token management.

    OwLiteSettings manages tokens and URLs within the OwLite system.
    It provides methods to retrieve and store tokens for authentication.

    Attributes:
        token_cache (Path): Path to store token information.
        urls_cache (Path): Path to store URL information.
    """

    def __init__(self) -> None:
        """Initialize OwLite settings.

        Initialize paths for OwLite cache directory to store tokens and URLs.
        """
        self.tokens_cache = OWLITE_CACHE_PATH / "tokens"
        self.urls_cache = OWLITE_CACHE_PATH / "urls"
        self.current_workspace_cache = OWLITE_CACHE_PATH / "current_workspace"
        self.connected_cache = OWLITE_CACHE_PATH / "connected"

    @property
    def tokens(self) -> Tokens | None:
        """Retrieve tokens or None if they don't exist.

        Returns:
            Tokens | None: An instance of Tokens representing the access token and refresh token,
            or None if the tokens don't exist.
        """
        read_tokens = read_text(self.tokens_cache)
        if not read_tokens:
            return None
        return Tokens.model_validate_json(read_tokens)

    @tokens.setter
    def tokens(self, new_tokens: Tokens | None) -> None:
        """Set new tokens or removes existing tokens.

        Args:
            new_tokens (Tokens | None): An instance of Tokens representing the new access token and refresh token.
            If None, existing tokens will be removed.
        """
        if new_tokens:
            write_text(self.tokens_cache, new_tokens.model_dump_json())
            log.debug(f"Saved access token='{new_tokens.access_token}' refresh token='{new_tokens.refresh_token}'")
        else:
            self.tokens_cache.unlink(missing_ok=True)

    @property
    def base_url(self) -> BaseURLs:
        """Retrieve base URLs.

        Returns the base URLs including FRONT, MAIN, and DOVE.
        If no custom URLs are set, it defaults to OwLite base URLs.

        Returns:
            BaseURLs: an instance of BaseURLs.
        """
        base_urls = read_text(self.urls_cache)
        if not base_urls:
            return BaseURLs()
        return BaseURLs.model_validate_json(base_urls)

    @base_url.setter
    def base_url(self, base_urls: BaseURLs) -> None:
        """Set or remove custom base URLs.

        Args:
            base_urls (BaseURLs): An instance of BaseURLs to set or remove custom base URLs.

        Raises:
            ValueError: If the provided 'base_urls' instance is invalid or incomplete.
        """
        write_text(self.urls_cache, base_urls.model_dump_json())

    @property
    def current_workspace(self) -> Workspace | None:
        """Retrieve current workspace.

        Returns:
            Workspace | None: The workspace, or None if it doesn't exist.
        """
        current_workspace = read_text(self.current_workspace_cache)
        if not current_workspace:
            return None
        return Workspace.model_validate_json(current_workspace)

    @current_workspace.setter
    def current_workspace(self, new_workspace: Workspace | None) -> None:
        """Set or remove current workspace.

        Args:
            new_workspace (Workspace | None): The workspace to set or None to remove the current workspace.
        """
        if new_workspace:
            write_text(self.current_workspace_cache, new_workspace.model_dump_json())
            log.debug(f"Saved current workspace: {new_workspace}")
        else:
            self.current_workspace_cache.unlink(missing_ok=True)


OWLITE_SETTINGS = OwLiteSettings()
