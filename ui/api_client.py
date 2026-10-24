from __future__ import annotations

from typing import Any

import requests


class ApiError(RuntimeError):
    pass


class OCRApiClient:
    def __init__(self, base_url: str, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    @property
    def origin(self) -> str:
        return self.base_url.removesuffix("/api")

    def asset_url(self, path: str | None) -> str | None:
        if not path:
            return None
        if path.startswith(("http://", "https://")):
            return path
        return f"{self.origin}{path}"

    def asset_bytes(self, path: str) -> bytes:
        response = self.session.get(self.asset_url(path), timeout=self.timeout)
        if not response.ok:
            self._payload(response)
        return response.content

    def get(self, path: str, *, timeout: int | None = None, **params: Any) -> Any:
        response = self.session.get(
            f"{self.base_url}{path}",
            params={key: value for key, value in params.items() if value is not None},
            timeout=timeout or self.timeout,
        )
        return self._payload(response)

    def post_json(self, path: str, payload: dict[str, Any], *, timeout: int | None = None) -> Any:
        response = self.session.post(
            f"{self.base_url}{path}",
            json=payload,
            timeout=timeout or self.timeout,
        )
        return self._payload(response)

    def patch_json(self, path: str, payload: dict[str, Any], *, timeout: int | None = None) -> Any:
        response = self.session.patch(
            f"{self.base_url}{path}",
            json=payload,
            timeout=timeout or self.timeout,
        )
        return self._payload(response)

    def post(self, path: str, *, timeout: int | None = None, **params: Any) -> Any:
        response = self.session.post(
            f"{self.base_url}{path}",
            params={key: value for key, value in params.items() if value is not None},
            timeout=timeout or self.timeout,
        )
        return self._payload(response)

    def delete(self, path: str, *, timeout: int | None = None, **params: Any) -> Any:
        response = self.session.delete(
            f"{self.base_url}{path}",
            params={key: value for key, value in params.items() if value is not None},
            timeout=timeout or self.timeout,
        )
        return self._payload(response)

    def download(self, path: str, *, timeout: int | None = None, **params: Any) -> tuple[bytes, str]:
        response = self.session.get(
            f"{self.base_url}{path}",
            params={key: value for key, value in params.items() if value is not None},
            timeout=timeout or self.timeout,
        )
        if not response.ok:
            self._payload(response)
        return response.content, response.headers.get("content-type", "application/octet-stream")

    def create_job(
        self,
        pages: list[tuple[str, bytes, str]],
        *,
        title: str,
        document_target: str,
        page_mode: str,
        preprocess: bool,
        max_tokens: int,
        model_override: str = "",
    ) -> dict[str, Any]:
        files = [("files", (name, data, content_type)) for name, data, content_type in pages]
        form = {
            "title": title,
            "document_target": document_target,
            "mode": page_mode,
            "preprocess": str(preprocess).lower(),
            "max_tokens": str(max_tokens),
        }
        if model_override:
            form["model"] = model_override
        response = self.session.post(
            f"{self.base_url}/jobs",
            files=files,
            data=form,
            timeout=1800,
        )
        payload = self._payload(response)
        if not isinstance(payload, dict):
            raise ApiError("Сервер вернул неожиданный формат задания.")
        return payload

    def recognize(
        self,
        *,
        filename: str,
        content_type: str,
        data: bytes,
        document_target: str,
        page_mode: str,
        preprocess: bool,
        max_tokens: int,
        model_override: str = "",
        save_intermediate_images: bool = True,
        return_coordinates: bool = True,
    ) -> dict[str, Any]:
        form = {"preprocess": str(preprocess).lower()}
        endpoint = "/recognize"
        if document_target == "page":
            endpoint = "/recognize/page"
            form.update(
                {
                    "mode": page_mode,
                    "max_tokens": str(max_tokens),
                    "save_intermediate_images": str(save_intermediate_images).lower(),
                    "return_coordinates": str(return_coordinates).lower(),
                }
            )
            if model_override:
                form["model"] = model_override
        response = self.session.post(
            f"{self.base_url}{endpoint}",
            files={"file": (filename, data, content_type)},
            data=form,
            timeout=1800,
        )
        payload = self._payload(response)
        if not isinstance(payload, dict):
            raise ApiError("Сервер вернул неожиданный формат результата.")
        return payload

    @staticmethod
    def _payload(response: requests.Response) -> Any:
        if response.ok:
            if response.status_code == 204 or not response.content:
                return None
            try:
                return response.json()
            except ValueError as exc:
                raise ApiError("Сервер вернул ответ без данных.") from exc
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = None
        raise ApiError(detail or f"Ошибка сервера: HTTP {response.status_code}.")
