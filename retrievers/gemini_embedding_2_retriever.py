from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
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
    Gemini Embedding 2 retriever using the Gemini API REST endpoint.

    Default model:
      - models/gemini-embedding-2-preview

    Required env:
      - GEMINI_API_KEY or GOOGLE_API_KEY

    Optional env:
      - GEMINI_EMBED_MODEL              default: models/gemini-embedding-2-preview
      - GEMINI_EMBED_BATCH_SIZE         default: 8
      - GEMINI_EMBED_TIMEOUT_S          default: 120
      - GEMINI_EMBED_MAX_RETRIES        default: 5
      - GEMINI_EMBED_RETRY_BASE_S       default: 1.5
      - GEMINI_OUTPUT_DIMENSIONALITY    default: unset
      - GEMINI_QUERY_TASK_TYPE          default: RETRIEVAL_QUERY
      - GEMINI_TARGET_TASK_TYPE         default: RETRIEVAL_DOCUMENT
      - GEMINI_QUERY_TEXT_TEMPLATE      default: unset
      - GEMINI_TARGET_TEXT_TEMPLATE     default: unset
      - GEMINI_NORMALIZE                default: 1
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.api_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or ""
        ).strip()
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY."
            )

        model = os.environ.get(
            "GEMINI_EMBED_MODEL",
            "models/gemini-embedding-2-preview",
        ).strip()
        self.model = model if model.startswith("models/") else f"models/{model}"

        self.batch_size = max(1, int(os.environ.get("GEMINI_EMBED_BATCH_SIZE", "8")))
        self.timeout_s = max(1.0, float(os.environ.get("GEMINI_EMBED_TIMEOUT_S", "120")))
        self.max_retries = max(1, int(os.environ.get("GEMINI_EMBED_MAX_RETRIES", "5")))
        self.retry_base_s = max(0.1, float(os.environ.get("GEMINI_EMBED_RETRY_BASE_S", "1.5")))

        out_dim = os.environ.get("GEMINI_OUTPUT_DIMENSIONALITY", "").strip()
        self.output_dimensionality: Optional[int] = int(out_dim) if out_dim else None

        self.query_task_type = os.environ.get(
            "GEMINI_QUERY_TASK_TYPE",
            "RETRIEVAL_QUERY",
        ).strip() or "RETRIEVAL_QUERY"
        self.target_task_type = os.environ.get(
            "GEMINI_TARGET_TASK_TYPE",
            "RETRIEVAL_DOCUMENT",
        ).strip() or "RETRIEVAL_DOCUMENT"
        self.query_text_template = os.environ.get("GEMINI_QUERY_TEXT_TEMPLATE", "").strip()
        self.target_text_template = os.environ.get("GEMINI_TARGET_TEXT_TEMPLATE", "").strip()
        self.do_normalize = _env_flag("GEMINI_NORMALIZE", True)

    def _uses_prompt_task_instructions(self) -> bool:
        model_name = self.model.rsplit("/", 1)[-1].lower()
        return model_name.startswith("gemini-embedding-2")

    @staticmethod
    def _apply_text_template(template: str, text: str) -> str:
        if "{content}" in template:
            return template.format(content=text)
        return f"{template} {text}".strip()

    def _format_text_for_task(self, text: str, *, is_query: bool) -> str:
        text = (text or "").strip()
        if not text:
            return text

        template = self.query_text_template if is_query else self.target_text_template
        if template:
            return self._apply_text_template(template, text)

        if not self._uses_prompt_task_instructions():
            return text

        if is_query:
            task_type = self.query_task_type.upper()
            prefix_map = {
                "RETRIEVAL_QUERY": "task: search result | query: {content}",
                "QUESTION_ANSWERING": "task: question answering | query: {content}",
                "FACT_CHECKING": "task: fact checking | query: {content}",
                "CODE_RETRIEVAL_QUERY": "task: code retrieval | query: {content}",
                "CLASSIFICATION": "task: classification | query: {content}",
                "CLUSTERING": "task: clustering | query: {content}",
                "SEMANTIC_SIMILARITY": "task: sentence similarity | query: {content}",
            }
            template = prefix_map.get(task_type, "task: search result | query: {content}")
            return template.format(content=text)

        return f"title: none | text: {text}"

    @staticmethod
    def _pil_to_png_bytes(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()

    def _load_image_bytes(self, key: Any) -> Optional[Tuple[bytes, str]]:
        image = getattr(key, "image", None)
        if isinstance(image, Image.Image):
            return self._pil_to_png_bytes(image), "image/png"

        img_path = (getattr(key, "img_path", "") or "").strip()
        if not img_path:
            return None
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        mime_type, _ = mimetypes.guess_type(img_path)
        if mime_type and mime_type.startswith("image/"):
            with open(img_path, "rb") as fh:
                return fh.read(), mime_type

        with Image.open(img_path) as img:
            return self._pil_to_png_bytes(img), "image/png"

    def _build_parts(self, key: Any, *, is_query: bool) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = []

        text = self._format_text_for_task((getattr(key, "text", "") or ""), is_query=is_query)
        if text:
            parts.append({"text": text})

        image_payload = self._load_image_bytes(key)
        if image_payload is not None:
            image_bytes, mime_type = image_payload
            parts.append(
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64.b64encode(image_bytes).decode("ascii"),
                    }
                }
            )

        if not parts:
            raise ValueError("Gemini retriever received an item with neither text nor image.")
        return parts

    def _build_request(self, key: Any, task_type: str, *, is_query: bool) -> Dict[str, Any]:
        request: Dict[str, Any] = {
            "model": self.model,
            "content": {"parts": self._build_parts(key, is_query=is_query)},
        }
        if not self._uses_prompt_task_instructions():
            request["taskType"] = task_type
        if self.output_dimensionality is not None:
            request["outputDimensionality"] = self.output_dimensionality
        return request

    def _request_with_retries(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"https://generativelanguage.googleapis.com/v1beta/{self.model}:batchEmbedContents"
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
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
                    f"Gemini batchEmbedContents failed with HTTP {exc.code}: {error_body[:1000]}"
                )
                if not retriable or attempt + 1 >= self.max_retries:
                    raise last_error from exc
            except urllib.error.URLError as exc:
                last_error = RuntimeError(f"Gemini batchEmbedContents failed: {exc}")
                if attempt + 1 >= self.max_retries:
                    raise last_error from exc

            time.sleep(self.retry_base_s * (2 ** attempt))

        if last_error is not None:
            raise last_error
        raise RuntimeError("Gemini batchEmbedContents failed without an explicit error.")

    def _embed_batch(self, keys: List[Any], task_type: str, *, is_query: bool) -> torch.Tensor:
        payload = {
            "requests": [
                self._build_request(key, task_type=task_type, is_query=is_query)
                for key in keys
            ],
        }
        response = self._request_with_retries(payload)
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list):
            raise RuntimeError(f"Unexpected Gemini response: {response}")

        rows: List[List[float]] = []
        for item in embeddings:
            values = item.get("values")
            if values is None and isinstance(item.get("embedding"), dict):
                values = item["embedding"].get("values")
            if not isinstance(values, list):
                raise RuntimeError(f"Missing embedding values in Gemini response item: {item}")
            rows.append([float(v) for v in values])

        tensor = torch.tensor(rows, dtype=torch.float32)
        if self.do_normalize and tensor.numel():
            tensor = F.normalize(tensor, p=2, dim=-1)
        return tensor.cpu()

    def _embed(self, keys: List[Any], task_type: str, *, is_query: bool) -> torch.Tensor:
        if not keys:
            return torch.empty((0, 0), dtype=torch.float32)

        outs: List[torch.Tensor] = []
        for batch in _chunked(keys, self.batch_size):
            outs.append(self._embed_batch(list(batch), task_type=task_type, is_query=is_query))
        return torch.cat(outs, dim=0)

    def embed_queries(self, keys: List[Any]) -> torch.Tensor:
        return self._embed(keys, task_type=self.query_task_type, is_query=True)

    def embed_targets(self, keys: List[Any]) -> torch.Tensor:
        return self._embed(keys, task_type=self.target_task_type, is_query=False)
