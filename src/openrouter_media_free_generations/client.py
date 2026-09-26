"""OpenRouter's dedicated image API, using only the Python standard library."""

from __future__ import annotations

import base64
import binascii
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_BASE_URL = "https://openrouter.ai"
_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/svg+xml": ".svg",
}


class OpenRouterError(RuntimeError):
    """An HTTP, transport, or malformed response error from OpenRouter."""


def _request(path: str, api_token: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        _BASE_URL + path,
        data=body,
        headers={
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST" if body is not None else "GET",
    )
    try:
        with urlopen(request, timeout=180) as response:
            result = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            error = json.loads(detail).get("error", detail)
            if isinstance(error, dict):
                detail = str(error.get("message", error))
            else:
                detail = str(error)
        except (ValueError, AttributeError):
            pass
        raise OpenRouterError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc
    except (ValueError, UnicodeDecodeError) as exc:
        raise OpenRouterError("OpenRouter returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise OpenRouterError("OpenRouter returned an unexpected response")
    return result


def _token(api_token: str | None) -> str:
    token = api_token or os.environ.get("OPENROUTER_API_KEY")
    if not token:
        raise ValueError("Pass api_token or set OPENROUTER_API_KEY")
    return token


@dataclass(frozen=True)
class GeneratedImage:
    data: bytes
    media_type: str = "image/png"

    def save(self, path: str | Path) -> Path:
        """Save bytes without converting the image format; add an extension if absent."""
        destination = Path(path)
        if not destination.suffix:
            destination = destination.with_suffix(_EXTENSIONS.get(self.media_type, ".bin"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.data)
        return destination


@dataclass(frozen=True)
class GeneratedImages:
    images: tuple[GeneratedImage, ...]
    usage: dict[str, Any]

    def save(self, path: str | Path, index: int = 0) -> Path:
        """Save one image from the response (the first by default)."""
        return self.images[index].save(path)


class ImageGenerator:
    """Generate images with a model slug and optional model-specific parameters."""

    def __init__(self, model_name: str, api_token: str | None = None, **model_parameters: Any):
        if not model_name:
            raise ValueError("model_name must not be empty")
        self.model_name = model_name
        self.api_token = _token(api_token)
        self.model_parameters = model_parameters
        self.last_result: GeneratedImages | None = None

    def generate(self, prompt: str, **parameters: Any) -> GeneratedImages:
        """Generate image(s). Per-call parameters override constructor parameters."""
        if not prompt:
            raise ValueError("prompt must not be empty")
        options = {**self.model_parameters, **parameters}
        if "model" in options or "prompt" in options:
            raise ValueError("model and prompt must be passed as their own arguments")
        if options.get("stream"):
            raise ValueError("stream=True is not supported by this simple client")
        response = _request(
            "/api/v1/images", self.api_token,
            {"model": self.model_name, "prompt": prompt, **options},
        )
        try:
            images = tuple(
                GeneratedImage(
                    base64.b64decode(item["b64_json"], validate=True),
                    item.get("media_type") or "image/png",
                )
                for item in response["data"]
            )
        except (KeyError, TypeError, binascii.Error) as exc:
            raise OpenRouterError("OpenRouter returned malformed image data") from exc
        if not images:
            raise OpenRouterError("OpenRouter returned no images")
        result = GeneratedImages(images, response.get("usage") or {})
        self.last_result = result
        return result

    def save(self, path: str | Path, index: int = 0) -> Path:
        """Save an image from the most recent generate() call."""
        if self.last_result is None:
            raise ValueError("Call generate() before save()")
        return self.last_result.save(path, index)


class ImageModelsList:
    """Fetch current image model metadata, including optional endpoint details."""

    def __init__(self, api_token: str | None = None):
        self.api_token = _token(api_token)

    def fetch(self, include_endpoints: bool = True) -> dict[str, Any]:
        """Return the raw model JSON, with per-provider details by default."""
        result = _request("/api/v1/images/models", self.api_token)
        models = result.get("data")
        if not isinstance(models, list):
            raise OpenRouterError("OpenRouter returned malformed model data")
        if include_endpoints:
            for model in models:
                path = model.get("endpoints")
                if not isinstance(path, str) or not path.startswith("/api/v1/images/models/") or not path.endswith("/endpoints"):
                    raise OpenRouterError("OpenRouter returned an invalid model endpoint path")
                model["endpoint_details"] = _request(path, self.api_token)
        return result

    def save_json(self, path: str | Path, include_endpoints: bool = True) -> Path:
        """Fetch current model metadata and save it as readable JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.fetch(include_endpoints), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return destination
