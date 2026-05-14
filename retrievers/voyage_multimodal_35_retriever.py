from __future__ import annotations

import base64
import io
import json
import logging
import mimetypes
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image


def _chunked(seq: Sequence[Any], batch_size: int):
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


class Retriever:
    """
    Voyage multimodal embedding retriever.

    This retriever prefers the official `voyageai` Python client when it is
    available and falls back to the Voyage REST API otherwise.

    Required env:
      - VOYAGE_API_KEY

    Optional env:
      - VOYAGE_MODEL              default: voyage-multimodal-3.5
      - VOYAGE_BATCH_SIZE         default: 8
      - VOYAGE_TRUNCATION         default: 1
      - VOYAGE_TIMEOUT_S          default: 120
      - VOYAGE_MAX_RETRIES        default: 5
      - VOYAGE_RETRY_BASE_S       default: 1.5
      - VOYAGE_OUTPUT_DIMENSION   default: unset
      - VOYAGE_QUERY_INPUT_TYPE   default: query
      - VOYAGE_TARGET_INPUT_TYPE  default: document
      - VOYAGE_NORMALIZE          default: 1
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.api_key = os.environ.get("VOYAGE_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("Voyage API key not found. Set VOYAGE_API_KEY.")

        self.model = os.environ.get("VOYAGE_MODEL", "voyage-multimodal-3.5").strip()
        self.batch_size = max(1, int(os.environ.get("VOYAGE_BATCH_SIZE", "8")))
        self.truncation = _env_flag("VOYAGE_TRUNCATION", True)
        self.timeout_s = max(1.0, float(os.environ.get("VOYAGE_TIMEOUT_S", "120")))
        self.max_retries = max(1, int(os.environ.get("VOYAGE_MAX_RETRIES", "5")))
        self.retry_base_s = max(0.1, float(os.environ.get("VOYAGE_RETRY_BASE_S", "1.5")))
        self.min_interval_s = max(0.0, float(os.environ.get("VOYAGE_MIN_INTERVAL_S", "0")))
        self.query_input_type = os.environ.get("VOYAGE_QUERY_INPUT_TYPE", "query").strip() or "query"
        self.target_input_type = os.environ.get("VOYAGE_TARGET_INPUT_TYPE", "document").strip() or "document"
        self.do_normalize = _env_flag("VOYAGE_NORMALIZE", True)
        self._request_lock = threading.Lock()
        self._last_request_started_at = 0.0
        self._last_embedding_dim: Optional[int] = None
        self._log = logging.getLogger("voyage_multimodal_35_retriever")

        out_dim = os.environ.get("VOYAGE_OUTPUT_DIMENSION", "").strip()
        self.output_dimension: Optional[int] = int(out_dim) if out_dim else None

        self._client = None
        try:
            import voyageai  # type: ignore

            self._client = voyageai.Client(api_key=self.api_key)
        except Exception:
            self._client = None

    @staticmethod
    def _pil_to_upload_payload(
        img: Image.Image,
        preferred_format: Optional[str] = None,
    ) -> Tuple[bytes, str]:
        if preferred_format == "jpeg":
            jpg_buf = io.BytesIO()
            img.convert("RGB").save(jpg_buf, format="JPEG", quality=95)
            return jpg_buf.getvalue(), "image/jpeg"

        png_buf = io.BytesIO()
        img.convert("RGB").save(png_buf, format="PNG")
        png_bytes = png_buf.getvalue()
        if len(png_bytes) <= 20 * 1024 * 1024:
            return png_bytes, "image/png"

        jpg_buf = io.BytesIO()
        img.convert("RGB").save(jpg_buf, format="JPEG", quality=95)
        return jpg_buf.getvalue(), "image/jpeg"

    def _load_image_payload(
        self,
        key: Any,
        preferred_format: Optional[str] = None,
    ) -> Optional[Tuple[bytes, str]]:
        image = getattr(key, "image", None)
        if isinstance(image, Image.Image):
            return self._pil_to_upload_payload(image, preferred_format=preferred_format)

        img_path = (getattr(key, "img_path", "") or "").strip()
        if not img_path:
            return None
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        if preferred_format is None:
            mime_type, _ = mimetypes.guess_type(img_path)
            if mime_type and mime_type.startswith("image/"):
                with open(img_path, "rb") as fh:
                    image_bytes = fh.read()
                if len(image_bytes) <= 20 * 1024 * 1024:
                    return image_bytes, mime_type

        with Image.open(img_path) as img:
            return self._pil_to_upload_payload(img, preferred_format=preferred_format)

    def _build_client_input(self, key: Any) -> List[Any]:
        content: List[Any] = []

        text = (getattr(key, "text", "") or "").strip()
        if text:
            content.append(text)

        image = getattr(key, "image", None)
        if isinstance(image, Image.Image):
            content.append(image.convert("RGB"))
        else:
            img_path = (getattr(key, "img_path", "") or "").strip()
            if img_path:
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Image not found: {img_path}")
                with Image.open(img_path) as img:
                    content.append(img.convert("RGB"))

        if not content:
            raise ValueError("Voyage retriever received an item with neither text nor image.")
        return content

    def _build_rest_input(
        self,
        key: Any,
        preferred_image_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        content: List[Dict[str, str]] = []

        text = (getattr(key, "text", "") or "").strip()
        if text:
            content.append({"type": "text", "text": text})

        image_payload = self._load_image_payload(key, preferred_format=preferred_image_format)
        if image_payload is not None:
            image_bytes, mime_type = image_payload
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            content.append(
                {
                    "type": "image_base64",
                    "image_base64": f"data:{mime_type};base64,{b64}",
                }
            )

        if not content:
            raise ValueError("Voyage retriever received an item with neither text nor image.")
        return {"content": content}

    def _request_with_retries(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = "https://api.voyageai.com/v1/multimodalembeddings"
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                retriable = exc.code in (429, 500, 502, 503, 504)
                last_error = RuntimeError(
                    f"Voyage multimodalembeddings failed with HTTP {exc.code}: {error_body[:1000]}"
                )
                if not retriable or attempt + 1 >= self.max_retries:
                    raise last_error from exc
                retry_after_s = 0.0
                try:
                    retry_after = exc.headers.get("Retry-After")
                    if retry_after:
                        retry_after_s = float(retry_after)
                except Exception:
                    retry_after_s = 0.0
            except urllib.error.URLError as exc:
                last_error = RuntimeError(f"Voyage multimodalembeddings failed: {exc}")
                if attempt + 1 >= self.max_retries:
                    raise last_error from exc
                retry_after_s = 0.0

            sleep_s = max(self.retry_base_s * (2 ** attempt), self.min_interval_s, retry_after_s)
            time.sleep(sleep_s)

        if last_error is not None:
            raise last_error
        raise RuntimeError("Voyage multimodalembeddings failed without an explicit error.")

    def _wait_for_slot(self) -> None:
        if self.min_interval_s <= 0:
            return
        with self._request_lock:
            now = time.monotonic()
            wait_s = (self._last_request_started_at + self.min_interval_s) - now
            if wait_s > 0:
                time.sleep(wait_s)
            self._last_request_started_at = time.monotonic()

    @staticmethod
    def _is_invalid_image_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "image file you have provided is invalid or corrupt" in msg or "invalid or corrupt" in msg

    def _response_to_tensor(self, response: Dict[str, Any]) -> torch.Tensor:
        embeddings = response.get("embeddings")
        if isinstance(embeddings, list):
            rows = embeddings
        else:
            data = response.get("data")
            if not isinstance(data, list):
                raise RuntimeError(f"Unexpected Voyage response: {response}")
            rows = []
            for item in data:
                if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                    raise RuntimeError(f"Unexpected Voyage response item: {item}")
                rows.append(item["embedding"])

        tensor = torch.tensor(rows, dtype=torch.float32)
        if tensor.numel():
            self._last_embedding_dim = int(tensor.shape[-1])
        if self.do_normalize and tensor.numel():
            tensor = F.normalize(tensor, p=2, dim=-1)
        return tensor.cpu()

    def _request_rest_tensor(
        self,
        keys: List[Any],
        input_type: str,
        preferred_image_format: Optional[str] = None,
    ) -> torch.Tensor:
        payload: Dict[str, Any] = {
            "inputs": [
                self._build_rest_input(key, preferred_image_format=preferred_image_format)
                for key in keys
            ],
            "model": self.model,
            "input_type": input_type,
            "truncation": self.truncation,
        }
        if self.output_dimension is not None:
            payload["output_dimension"] = self.output_dimension

        response = self._request_with_retries(payload)
        return self._response_to_tensor(response)

    def _recover_invalid_image_batch(self, keys: List[Any], input_type: str) -> torch.Tensor:
        recovered_rows: List[Optional[torch.Tensor]] = []
        bad_paths: List[str] = []
        dim = self._last_embedding_dim

        for idx, key in enumerate(keys):
            try:
                row = self._request_rest_tensor([key], input_type=input_type)
                recovered_rows.append(row)
                if row.numel():
                    dim = int(row.shape[-1])
                continue
            except Exception as exc:
                if not self._is_invalid_image_error(exc):
                    raise

            try:
                row = self._request_rest_tensor(
                    [key],
                    input_type=input_type,
                    preferred_image_format="jpeg",
                )
                recovered_rows.append(row)
                if row.numel():
                    dim = int(row.shape[-1])
                continue
            except Exception as exc:
                if not self._is_invalid_image_error(exc):
                    raise

            path = (getattr(key, "img_path", "") or "").strip() or f"<item {idx}>"
            bad_paths.append(path)
            recovered_rows.append(None)

        if dim is None:
            raise RuntimeError(
                "Voyage rejected every item in a batch with invalid-image errors and no embedding "
                "dimension is available for zero-fill recovery."
            )

        zero_row = torch.zeros((1, dim), dtype=torch.float32)
        rows = [row if row is not None else zero_row.clone() for row in recovered_rows]
        self._log.warning(
            "Voyage rejected %d image(s) after single-item retries; using zero vectors for: %s",
            len(bad_paths),
            ", ".join(bad_paths[:8]),
        )
        return torch.cat(rows, dim=0)

    def _embed_with_client(self, keys: List[Any], input_type: str) -> torch.Tensor:
        assert self._client is not None

        kwargs: Dict[str, Any] = {
            "inputs": [self._build_client_input(key) for key in keys],
            "model": self.model,
            "input_type": input_type,
            "truncation": self.truncation,
        }
        if self.output_dimension is not None:
            kwargs["output_dimension"] = self.output_dimension

        result = self._client.multimodal_embed(**kwargs)
        tensor = torch.tensor(result.embeddings, dtype=torch.float32)
        if tensor.numel():
            self._last_embedding_dim = int(tensor.shape[-1])
        if self.do_normalize and tensor.numel():
            tensor = F.normalize(tensor, p=2, dim=-1)
        return tensor.cpu()

    def _embed_with_rest(self, keys: List[Any], input_type: str) -> torch.Tensor:
        try:
            return self._request_rest_tensor(keys, input_type=input_type)
        except Exception as exc:
            if not self._is_invalid_image_error(exc):
                raise
            return self._recover_invalid_image_batch(keys, input_type=input_type)

    def _embed(self, keys: List[Any], input_type: str) -> torch.Tensor:
        if not keys:
            return torch.empty((0, 0), dtype=torch.float32)

        outs: List[torch.Tensor] = []
        for batch in _chunked(keys, self.batch_size):
            batch_list = list(batch)
            self._wait_for_slot()
            if self._client is not None:
                try:
                    outs.append(self._embed_with_client(batch_list, input_type=input_type))
                    continue
                except Exception:
                    pass
            outs.append(self._embed_with_rest(batch_list, input_type=input_type))
        return torch.cat(outs, dim=0)

    def embed_queries(self, keys: List[Any]) -> torch.Tensor:
        return self._embed(keys, input_type=self.query_input_type)

    def embed_targets(self, keys: List[Any]) -> torch.Tensor:
        return self._embed(keys, input_type=self.target_input_type)
