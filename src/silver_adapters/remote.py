import asyncio
from typing import Any, Dict, List


class RemoteTrainingClient:
    def __init__(
        self, base_url: str, headers: Dict[str, str] = None, session: Any = None,
        timeout: float = 30.0,
    ):
        if not base_url.strip():
            raise ValueError("Remote training base URL is required")
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self._session = session
        if timeout <= 0:
            raise ValueError("Remote training timeout must be positive")
        self.timeout = timeout
        try:
            import requests

            self._requests = requests
        except ImportError as error:
            raise ImportError(
                "requests package is required for RemoteTrainingClient"
            ) from error

    def _get_session(self):
        if self._session:
            return self._session
        return self._requests.Session()

    async def start(self, payload: Any) -> Dict[str, Any]:
        return await self._send("/runs", "POST", payload)

    async def events(self, run_id: str) -> List[Dict[str, Any]]:
        return await self._send(f"/runs/{run_id}/events", "GET")

    async def stop(
        self, run_id: str, reason: str = "stopped by user"
    ) -> Dict[str, Any]:
        return await self._send(f"/runs/{run_id}/stop", "POST", {"reason": reason})

    async def checkpoint(self, run_id: str) -> Any:
        return await self._send(f"/runs/{run_id}/checkpoint", "GET")

    async def _send(self, path: str, method: str, body: Any = None) -> Any:
        def sync_request():
            session = self._get_session()
            response = session.request(
                method,
                f"{self.base_url}{path}",
                headers={"content-type": "application/json", **self.headers},
                json=body if body is not None else None,
                timeout=self.timeout,
            )
            if not response.ok:
                raise RuntimeError(
                    f"Remote training request failed: {response.status_code} {response.text}"
                )
            return response.json()

        return await asyncio.to_thread(sync_request)
