# qwen3vl_vllm_retriever.py
'''
Docstring per retrievers.qwen3vl2b_vllm_retriever
env gte-qwen-retriever
'''


from __future__ import annotations

import multiprocessing as mp
mp.set_start_method("spawn", force=True)


import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from vllm import LLM, EngineArgs
from vllm.multimodal.utils import fetch_image



def _format_input_to_conversation(input_dict: Dict[str, Any], instruction: str) -> List[Dict]:
    content = []

    text = input_dict.get("text")
    image = input_dict.get("image")

    if image:
        # vLLM accepts: URL, oss, or file://path
        if isinstance(image, str):
            if image.startswith(("http", "https", "oss")):
                image_content = image
            else:
                abs_image_path = os.path.abspath(image)
                image_content = "file://" + abs_image_path
        else:
            image_content = image

        content.append({"type": "image", "image": image_content})

    if text:
        content.append({"type": "text", "text": text})

    if not content:
        content.append({"type": "text", "text": ""})

    return [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content},
    ]


def _prepare_vllm_input(input_dict: Dict[str, Any], llm: LLM, instruction: str) -> Dict[str, Any]:
    conversation = _format_input_to_conversation(input_dict, instruction)

    prompt_text = llm.llm_engine.tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )

    image = input_dict.get("image")
    multi_modal_data = None

    if image:
        if isinstance(image, str):
            if image.startswith(("http", "https", "oss")):
                try:
                    image_obj = fetch_image(image)
                    multi_modal_data = {"image": image_obj}
                except Exception:
                    multi_modal_data = None
            else:
                abs_path = os.path.abspath(image)
                if os.path.exists(abs_path):
                    img = Image.open(abs_path).convert("RGB")
                    multi_modal_data = {"image": img}
        else:
            multi_modal_data = {"image": image}

    return {"prompt": prompt_text, "multi_modal_data": multi_modal_data}


class Retriever:
    """
    EmbeddingRetriever for Qwen3-VL-Embedding-* via vLLM (runner='pooling').

    Usage with your evaluator:
      --retriever_module qwen3vl_vllm_retriever --retriever_class Retriever
      optionally set env vars to control model path, dtype, and instructions.

    Retriever args (passed by evaluator as Retriever(device=...)):
      - device: "cuda" or "cpu" (vLLM will use available GPU if installed)

    Configure via environment variables (simple, no evaluator changes needed):
      QWEN3VL_MODEL_PATH   (default: "models/Qwen3-VL-Embedding-8B")
      QWEN3VL_DTYPE        (default: "bfloat16")
      QWEN3VL_MAX_LENGTH   (default: "512")
      QWEN3VL_Q_INSTR      (default: "Represent the user's input.")
      QWEN3VL_T_INSTR      (default: "Represent the user's input.")
      QWEN3VL_BATCH_SIZE   (default: "32")
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

        self.model_path = os.environ.get("QWEN3VL_MODEL_PATH", "Qwen/Qwen3-VL-Embedding-2B")
        self.dtype = os.environ.get("QWEN3VL_DTYPE", "bfloat16")
        self.max_length = int(os.environ.get("QWEN3VL_MAX_LENGTH", "512"))
        self.q_instruction = os.environ.get("Q_INSTRUCTION", "Represent the user's input.")
        self.t_instruction = os.environ.get("QWEN3VL_T_INSTR", "Represent the user's input.")
        self.batch_size = int(os.environ.get("QWEN3VL_BATCH_SIZE", "32"))

        # engine_args = EngineArgs(
        #     model=self.model_path,
        #     runner="pooling",
        #     dtype=self.dtype,
        #     trust_remote_code=True,
        # )

        gpu_mem_util = float(os.environ.get("QWEN3VL_GPU_MEM_UTIL", "0.70"))
        max_model_len = int(os.environ.get("QWEN3VL_MAX_MODEL_LEN", "0"))

        ea_kwargs = dict(
            model=self.model_path,
            runner="pooling",
            dtype=self.dtype,
            limit_mm_per_prompt={"image": 1},
            enforce_eager=True,  # avoids some compiled/flash paths and is more stable
            mm_encoder_attn_backend="TORCH_SDPA",
            gpu_memory_utilization=gpu_mem_util,
        )
        if max_model_len > 0:
            ea_kwargs["max_model_len"] = max_model_len

        engine_args = EngineArgs(**ea_kwargs)

        # vLLM decides device based on installation; on GPU systems this will use CUDA.
        self.llm = LLM(**vars(engine_args))

    # def _keys_to_inputs(self, keys: List, instruction: str) -> List[Dict[str, Any]]:
    #     inputs = []
    #     for k in keys:
    #         text = (getattr(k, "text", "") or "").strip()
    #         img_path = (getattr(k, "img_path", "") or "").strip()

    #         inp: Dict[str, Any] = {}
    #         if text:
    #             inp["text"] = text
    #         if img_path:
    #             inp["image"] = img_path

    #         inputs.append(_prepare_vllm_input(inp, self.llm, instruction))
    #     return inputs

    def _keys_to_inputs(self, keys: List, instruction: str) -> List[Dict[str, Any]]:
        inputs = []

        for k in keys:
            text = (getattr(k, "text", "") or "").strip()
            img_obj = getattr(k, "image", None)
            img_path = (getattr(k, "img_path", "") or "").strip()

            inp: Dict[str, Any] = {}

            if text:
                inp["text"] = text

            # ✅ PREFER in-memory PIL image (attacked)
            if img_obj is not None:
                inp["image"] = img_obj

            # 🔁 fallback: load from disk only if no image object
            elif img_path:
                inp["image"] = img_path

            inputs.append(_prepare_vllm_input(inp, self.llm, instruction))

        return inputs

    def _embed(self, vllm_inputs: List[Dict[str, Any]]) -> torch.Tensor:
        # vLLM returns embeddings as python lists / numpy arrays
        outs = self.llm.embed(vllm_inputs)

        embs = []
        for o in outs:
            emb = o.outputs.embedding  # list[float]
            embs.append(emb)

        arr = np.asarray(embs, dtype=np.float32)  # [N,D]
        return torch.from_numpy(arr)  # CPU tensor

    def embed_queries(self, keys: List) -> torch.Tensor:
        vllm_inputs = self._keys_to_inputs(keys, instruction=self.q_instruction)
        # optional micro-batching to control memory
        out_chunks = []
        for i in range(0, len(vllm_inputs), self.batch_size):
            out_chunks.append(self._embed(vllm_inputs[i : i + self.batch_size]))
        return torch.cat(out_chunks, dim=0)

    def embed_targets(self, keys: List) -> torch.Tensor:
        vllm_inputs = self._keys_to_inputs(keys, instruction=self.t_instruction)
        out_chunks = []
        for i in range(0, len(vllm_inputs), self.batch_size):
            out_chunks.append(self._embed(vllm_inputs[i : i + self.batch_size]))
        return torch.cat(out_chunks, dim=0)
