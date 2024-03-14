from typing import Any, Callable, Optional, Union

import requests

from .cache.tokens import Tokens
from .constants import OWLITE_API_DEFAULT_TIMEOUT
from .logger import log
from .owlite_settings import OWLITE_SETTINGS

ResponseType = Union[dict, int, str, bool, list[dict]]


class APIBase:
    """Represents a base class for making HTTP requests to the OwLite API."""

    def __init__(self, base_url: str, name: str) -> None:
        self.name = name
        self._base_url = base_url.rstrip("/")
        self.default_timeout = OWLITE_API_DEFAULT_TIMEOUT

    @property
    def base_url(self) -> str:  # pylint: disable=missing-function-docstring
        return self._base_url

    @base_url.setter
    def base_url(self, new_base_url: str) -> None:
        self._base_url = new_base_url.rstrip("/")

    def _create_request_kwargs(self, **kwargs: dict[str, Any]) -> dict:
        """Creates request keyword arguments with default settings and authentication headers.

        Returns:
            dict: Request keyword arguments.
        """
        ret = {"timeout": self.default_timeout}
        ret.update(kwargs)  # type: ignore

        tokens = OWLITE_SETTINGS.tokens
        if tokens is not None:
            headers: dict[str, Any] = ret.get("headers", {})  # type: ignore
            auth = {"Authorization": "Bearer " + tokens.access_token}
            headers.update(auth)
            ret["headers"] = headers  # type: ignore

        return ret

    def _request(
        self, request_callable: Callable[[], requests.Response], num_retry_after_timeout: int = 0
    ) -> ResponseType:
        """Sends an HTTP request and handles response and retries on timeout or authorization issues.

        Args:
            request_callable: The callable to make the request.
            num_retry_after_timeout (int): Number of retries after timeout. Defaults to 0.

        Returns:
            ResponseType: Response data.

        Raises:
            Timeout: When request times out.
            HTTPError: When the request fails.
        """
        i = 0
        while i < num_retry_after_timeout + 1:
            try:
                response = request_callable()
                if response.ok:  # response with status code 200 ~ 399
                    log.debug(f"Request succeeded with status {response.status_code}")

                    return response.json()

                if response.status_code == 401:  # access token expired
                    log.debug("Access token expired, attempting refresh")

                    try:  # attempt refresh
                        tokens = OWLITE_SETTINGS.tokens
                        assert tokens is not None

                        resp = requests.post(
                            f"{OWLITE_SETTINGS.base_url.MAIN}/login/refresh",
                            json={"refresh_token": tokens.refresh_token},
                            timeout=self.default_timeout,
                            headers={"Authorization": "Bearer " + tokens.access_token},
                        )
                        log.debug(f"Token refresh request : {resp.status_code}")
                        if not resp.ok:
                            resp.raise_for_status()

                        refresh_res = resp.json()
                        assert isinstance(refresh_res, dict)
                        OWLITE_SETTINGS.tokens = Tokens(
                            access_token=refresh_res["access_token"], refresh_token=refresh_res["refresh_token"]
                        )  # token refreshed

                        log.debug("Token refreshed, re-attempting original request")
                        i -= 1
                        continue

                    except Exception as e:  # refresh failed, force to login again
                        log.error("Login session expired. Please log in again using 'owlite login'")  # UX
                        OWLITE_SETTINGS.tokens = None
                        raise e

                log.debug(
                    f"Request failed with status {response.status_code}, raising HTTPError\n"
                    f"Response content was:\n{response.content!r}"
                )
                response.raise_for_status()

            except requests.exceptions.Timeout:  # request timeout
                i += 1
                continue

        raise requests.exceptions.Timeout()

    def get(self, url: str, params: Optional[dict[str, Any]] = None, **kwargs: dict[str, Any]) -> ResponseType:
        """Sends a GET request to the given URL.

        Args:
            url (str): URL endpoint.
            params (dict, optional): Parameters for the request. Defaults to None.

        Returns:
            ResponseType: Response data.

        Raises:
            Timeout: When request times out.
            HTTPError: When the request fails.
        """
        log.debug(f"GET {self.base_url + url}")

        def request_callable() -> requests.Response:
            request_kwargs = self._create_request_kwargs(**kwargs)
            return requests.get(url=self.base_url + url, params=params, **request_kwargs)

        return self._request(request_callable)

    def post(self, url: str, data: Optional[dict[str, Any]] = None, **kwargs: dict[str, Any]) -> ResponseType:
        """Sends a POST request to the given URL.

        Args:
            url (str): URL endpoint.
            data (dict, optional): Data to be sent. Defaults to None.

        Returns:
            ResponseType: Response data.

        Raises:
            Timeout: When request times out.
            HTTPError: When the request fails.
        """
        log.debug(f"POST {self.base_url + url}")

        def request_callable() -> requests.Response:
            request_kwargs = self._create_request_kwargs(**kwargs)
            return requests.post(url=self.base_url + url, data=data, **request_kwargs)

        return self._request(request_callable)

    def put(self, url: str, data: Optional[dict[str, Any]] = None, **kwargs: dict[str, Any]) -> ResponseType:
        """Sends a PUT request to the given URL.

        Args:
            url (str): URL endpoint.
            data (dict, optional): Data to be sent. Defaults to None.

        Returns:
            ResponseType: Response data.

        Raises:
            Timeout: When request times out.
            HTTPError: When the request fails.
        """
        log.debug(f"PUT {self.base_url + url}")

        def request_callable() -> requests.Response:
            request_kwargs = self._create_request_kwargs(**kwargs)
            return requests.put(url=self.base_url + url, data=data, **request_kwargs)

        return self._request(request_callable)

    def patch(self, url: str, data: Optional[dict[str, Any]] = None, **kwargs: dict[str, Any]) -> ResponseType:
        """Sends a PATCH request to the given URL.

        Args:
            url (str): URL endpoint.
            data (dict, optional): Data to be sent. Defaults to None.

        Returns:
            ResponseType: Response data.

        Raises:
            Timeout: When request times out.
            HTTPError: When the request fails.
        """
        log.debug(f"PATCH {self.base_url + url}")

        def request_callable() -> requests.Response:
            request_kwargs = self._create_request_kwargs(**kwargs)
            return requests.patch(url=self.base_url + url, data=data, **request_kwargs)

        return self._request(request_callable)

    def delete(self, url: str, data: Optional[dict[str, Any]] = None, **kwargs: dict[str, Any]) -> ResponseType:
        """Sends a DELETE request to the given URL.

        Args:
            url (str): URL endpoint.
            data (dict, optional): Data to be sent. Defaults to None.

        Returns:
            ResponseType: Response data.

        Raises:
            Timeout: When request times out.
            HTTPError: When the request fails.
        """
        log.debug(f"DELETE {self.base_url + url}")

        def request_callable() -> requests.Response:
            request_kwargs = self._create_request_kwargs(**kwargs)
            return requests.delete(url=self.base_url + url, data=data, **request_kwargs)

        return self._request(request_callable)


MAIN_API_BASE: APIBase = APIBase(OWLITE_SETTINGS.base_url.MAIN, "OWLITE_MAIN_API_BASE")
DOVE_API_BASE: APIBase = APIBase(OWLITE_SETTINGS.base_url.DOVE, "OWLITE_DOVE_API_BASE")
