"""Shared OpenAI-compatible NVIDIA VLM transport with no media logging."""
from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from typing import Any

import cv2
import numpy as np


class NvidiaVLMError(RuntimeError):
    """Sanitized remote failure containing no credentials or payload."""
    def __init__(self, message: str, status: int | None = None,
                 retryable: bool | None = None):
        super().__init__(message)
        self.status = status
        self.retryable = (status is None or status in (408, 425, 429) or status >= 500
                          if retryable is None else bool(retryable))


def _sanitized_http_error(exc: urllib.error.HTTPError) -> str:
    """Return only NVIDIA's bounded error message, never its response envelope."""
    label = f"HTTP {exc.code}"
    try:
        body = json.loads(exc.read().decode("utf-8", errors="replace"))
        detail = body.get("error", {}).get("message")
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return label
    if not isinstance(detail, str):
        return label
    detail = re.sub(r"\s+", " ", detail).strip()[:300]
    return f"{label}: {detail}" if detail else label


class NvidiaVLMClient:
    """Encode in memory and call an NVIDIA OpenAI-compatible chat endpoint."""
    def __init__(self, api_key: str, endpoint: str, model: str,
                 timeout: float = 25, max_image_dim: int = 1024,
                 jpeg_quality: int = 85):
        self._key, self.endpoint, self.model = api_key, endpoint, model
        self.timeout, self.max_image_dim, self.jpeg_quality = timeout, max_image_dim, jpeg_quality

    def encode(self, frame: np.ndarray, *, max_image_dim: int | None = None,
               jpeg_quality: int | None = None) -> bytes:
        """Resize and JPEG-encode one frame without touching disk."""
        max_dim = self.max_image_dim if max_image_dim is None else max(64, int(max_image_dim))
        quality = max(1, min(100, self.jpeg_quality if jpeg_quality is None
                             else int(jpeg_quality)))
        h, w = frame.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                               interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", frame,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise NvidiaVLMError("JPEG encoding failed")
        return encoded.tobytes()

    def request(self, prompt: str, images: list[bytes], max_tokens: int = 700,
                response_format: dict[str, Any] | None = None,
                timeout: float | None = None) -> Any:
        """Submit text plus still/multi-frame input and return message content."""
        content: list[dict] = [{"type": "text", "text": prompt}]
        content.extend({"type": "image_url", "image_url": {"url":
                       "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")}}
                       for image in images)
        payload = {"model": self.model, "messages": [{"role": "user", "content": content}],
                   "temperature": 0.1, "max_tokens": max_tokens}
        if response_format is not None:
            payload["response_format"] = response_format
        req = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {self._key}",
                                              "Content-Type": "application/json",
                                              "Accept": "application/json"})
        try:
            with urllib.request.urlopen(
                    req, timeout=self.timeout if timeout is None else max(0.1, float(timeout))) as response:
                body = json.loads(response.read().decode())
            return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            retryable = exc.code in (408, 425, 429) or exc.code >= 500
            raise NvidiaVLMError(_sanitized_http_error(exc), exc.code, retryable) from exc
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, IndexError,
                ValueError, json.JSONDecodeError) as exc:
            raise NvidiaVLMError(type(exc).__name__) from exc
