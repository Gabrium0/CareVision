"""Shared OpenAI-compatible NVIDIA VLM transport with no media logging."""
from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import Any

import cv2
import numpy as np


class NvidiaVLMError(RuntimeError):
    """Sanitized remote failure containing no credentials or payload."""
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class NvidiaVLMClient:
    """Encode in memory and call an NVIDIA OpenAI-compatible chat endpoint."""
    def __init__(self, api_key: str, endpoint: str, model: str,
                 timeout: float = 25, max_image_dim: int = 1024,
                 jpeg_quality: int = 85):
        self._key, self.endpoint, self.model = api_key, endpoint, model
        self.timeout, self.max_image_dim, self.jpeg_quality = timeout, max_image_dim, jpeg_quality

    def encode(self, frame: np.ndarray) -> bytes:
        """Resize and JPEG-encode one frame without touching disk."""
        h, w = frame.shape[:2]
        if max(h, w) > self.max_image_dim:
            scale = self.max_image_dim / max(h, w)
            frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                               interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", frame,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise NvidiaVLMError("JPEG encoding failed")
        return encoded.tobytes()

    def request(self, prompt: str, images: list[bytes], max_tokens: int = 700) -> Any:
        """Submit text plus still/multi-frame input and return message content."""
        content: list[dict] = [{"type": "text", "text": prompt}]
        content.extend({"type": "image_url", "image_url": {"url":
                       "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")}}
                       for image in images)
        payload = {"model": self.model, "messages": [{"role": "user", "content": content}],
                   "temperature": 0.1, "max_tokens": max_tokens}
        req = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {self._key}",
                                              "Content-Type": "application/json",
                                              "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = json.loads(response.read().decode())
            return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            raise NvidiaVLMError(f"HTTP {exc.code}", exc.code) from exc
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, IndexError,
                ValueError, json.JSONDecodeError) as exc:
            raise NvidiaVLMError(type(exc).__name__) from exc
