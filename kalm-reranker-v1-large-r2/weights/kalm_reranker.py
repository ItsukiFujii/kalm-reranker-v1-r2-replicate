from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from .kalm_reranker_utils import (
        DEFAULT_INSTRUCTION,
        DEFAULT_SYSTEM_INSTRUCTION,
        answer_token_id,
        build_decoder_text,
        cast_floating_parameters,
        extract_yes_no_logits,
        forward_reranker_model,
        get_encoder,
        pool_encoder_chunks,
        validate_text_pairs,
    )
except ImportError:  # Support ``from kalm_reranker import KaLMReranker``.
    from kalm_reranker_utils import (
        DEFAULT_INSTRUCTION,
        DEFAULT_SYSTEM_INSTRUCTION,
        answer_token_id,
        build_decoder_text,
        cast_floating_parameters,
        extract_yes_no_logits,
        forward_reranker_model,
        get_encoder,
        pool_encoder_chunks,
        validate_text_pairs,
    )


class KaLMReranker:
    """Score query-document relevance with a KaLM encoder-decoder reranker.

    The returned score is ``P(yes)`` after applying a two-class softmax to the
    model's ``yes`` and ``no`` logits.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        batch_size: int = 32,
        query_max_length: int = 512,
        max_length: int = 1024,
        chunk_size: Optional[int] = 4,
        instruction: str = DEFAULT_INSTRUCTION,
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
        **model_kwargs: Any,
    ) -> None:
        if not isinstance(model_name_or_path, str) or not model_name_or_path:
            raise ValueError("model_name_or_path must be a non-empty string.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if query_max_length <= 0 or max_length <= 0:
            raise ValueError("query_max_length and max_length must be positive.")
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be positive or None.")
        if not isinstance(instruction, str) or not isinstance(system_instruction, str):
            raise TypeError("instruction and system_instruction must be strings.")

        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype, self.device)
        self.batch_size = batch_size
        self.query_max_length = query_max_length
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.instruction = instruction
        self.system_instruction = system_instruction

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError(
                    "The tokenizer must define a pad token or an EOS token."
                )
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Final decoder-token indexing assumes right padding, matching training.
        self.tokenizer.padding_side = "right"

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            dtype=self.dtype,
            **model_kwargs,
        )
        cast_floating_parameters(self.model, self.dtype)
        self.model.to(device=self.device)
        self.model.eval()

        self.yes_token_id = self._answer_token_id("yes")
        self.no_token_id = self._answer_token_id("no")

    @staticmethod
    def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        return resolved

    @staticmethod
    def _resolve_dtype(
        dtype: Optional[Union[str, torch.dtype]], device: torch.device
    ) -> torch.dtype:
        if dtype is None:
            return torch.bfloat16 if device.type == "cuda" else torch.float32
        if isinstance(dtype, torch.dtype):
            return dtype
        if not isinstance(dtype, str):
            raise TypeError(
                "dtype must be a torch.dtype or a string such as 'bfloat16'."
            )
        normalized = dtype.lower().removeprefix("torch.")
        supported = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if normalized not in supported:
            raise ValueError(f"Unsupported dtype: {dtype!r}.")
        return supported[normalized]

    def _answer_token_id(self, answer: str) -> int:
        return answer_token_id(self.tokenizer, answer)

    def _get_encoder(self):
        return get_encoder(self.model)

    @staticmethod
    def _pool_encoder_chunks(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return pool_encoder_chunks(hidden_states, attention_mask, chunk_size)

    def _decoder_text(self, query: str, instruction: str) -> str:
        return build_decoder_text(
            self.tokenizer,
            query,
            instruction,
            self.system_instruction,
            self.query_max_length,
        )

    @staticmethod
    def _validate_pairs(
        pairs: Sequence[Tuple[str, str]],
    ) -> List[Tuple[str, str]]:
        return validate_text_pairs(pairs)

    @torch.inference_mode()
    def _predict_batch(
        self, pairs: Sequence[Tuple[str, str]], instruction: str
    ) -> List[float]:
        encoder_texts = [f"<Document>: {document}" for _, document in pairs]
        decoder_texts = [self._decoder_text(query, instruction) for query, _ in pairs]

        encoder_batch = self.tokenizer(
            encoder_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)
        decoder_batch = self.tokenizer(
            decoder_texts,
            padding=True,
            pad_to_multiple_of=8,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)

        outputs = forward_reranker_model(
            self.model,
            input_ids=encoder_batch["input_ids"],
            attention_mask=encoder_batch["attention_mask"],
            decoder_input_ids=decoder_batch["input_ids"],
            decoder_attention_mask=decoder_batch["attention_mask"],
            encoder_chunk_size=self.chunk_size,
        )
        yes_no_logits = extract_yes_no_logits(
            outputs.logits,
            decoder_batch["attention_mask"],
            self.yes_token_id,
            self.no_token_id,
        )
        return torch.softmax(yes_no_logits, dim=-1)[:, 0].cpu().tolist()

    def predict(
        self,
        pairs: Sequence[Tuple[str, str]],
        *,
        instruction: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> List[float]:
        """Return ``P(yes)`` scores in the same order as ``pairs``."""
        validated_pairs = self._validate_pairs(pairs)
        if not validated_pairs:
            return []
        effective_instruction = self.instruction if instruction is None else instruction
        if not isinstance(effective_instruction, str):
            raise TypeError("instruction must be a string or None.")
        effective_batch_size = self.batch_size if batch_size is None else batch_size
        if not isinstance(effective_batch_size, int) or effective_batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        length_sorted_indices = np.argsort(
            [-(len(query) + len(document)) for query, document in validated_pairs]
        )
        sorted_pairs = [validated_pairs[index] for index in length_sorted_indices]

        tested_batch_size = effective_batch_size
        first_batch_scores: Optional[List[float]] = None
        while tested_batch_size > 1:
            try:
                first_batch_scores = self._predict_batch(
                    sorted_pairs[: min(len(sorted_pairs), tested_batch_size)],
                    effective_instruction,
                )
                break
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                tested_batch_size = max(1, tested_batch_size * 3 // 4)

        if first_batch_scores is None:
            sorted_scores: List[float] = []
            loop_start = 0
        else:
            sorted_scores = list(first_batch_scores)
            loop_start = tested_batch_size
        try:
            for start in range(loop_start, len(sorted_pairs), tested_batch_size):
                sorted_scores.extend(
                    self._predict_batch(
                        sorted_pairs[start : start + tested_batch_size],
                        effective_instruction,
                    )
                )
        except torch.cuda.OutOfMemoryError as error:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA ran out of memory during reranking. Retry with a smaller "
                "batch_size or shorter max_length."
            ) from error
        inverse_indices = np.argsort(length_sorted_indices)
        return [sorted_scores[index] for index in inverse_indices]

    def rank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: Optional[str] = None,
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Union[int, float]]]:
        """Rank documents and return ``corpus_id``/``score`` dictionaries."""
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
            raise TypeError("documents must be a sequence of strings.")
        if any(not isinstance(document, str) for document in documents):
            raise TypeError("every document must be a string.")
        if top_k is not None and (not isinstance(top_k, int) or top_k < 0):
            raise ValueError("top_k must be a non-negative integer or None.")

        scores = self.predict(
            [(query, document) for document in documents],
            instruction=instruction,
            batch_size=batch_size,
        )
        rankings: List[Dict[str, Union[int, float]]] = [
            {"corpus_id": corpus_id, "score": score}
            for corpus_id, score in enumerate(scores)
        ]
        rankings.sort(key=lambda item: item["score"], reverse=True)
        return rankings if top_k is None else rankings[:top_k]


__all__ = ["KaLMReranker"]
