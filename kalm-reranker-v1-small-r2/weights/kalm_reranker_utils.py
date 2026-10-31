from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput

DEFAULT_INSTRUCTION = "Given a query, retrieve documents that answer the query."
DEFAULT_SYSTEM_INSTRUCTION = (
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
)


def validate_text_pairs(inputs: Sequence[Sequence[str]]) -> list[tuple[str, str]]:
    """Validate and normalize a batch of ``(query, document)`` pairs."""
    if isinstance(inputs, (str, bytes)) or not isinstance(inputs, Sequence):
        raise TypeError("inputs must be a sequence of (query, document) pairs.")

    validated: list[tuple[str, str]] = []
    for index, pair in enumerate(inputs):
        if (
            isinstance(pair, (str, bytes))
            or not isinstance(pair, Sequence)
            or len(pair) != 2
        ):
            raise ValueError(f"inputs[{index}] must contain exactly two strings.")
        query, document = pair
        if not isinstance(query, str) or not isinstance(document, str):
            raise TypeError(f"inputs[{index}] must contain exactly two strings.")
        validated.append((query, document))
    return validated


def answer_token_id(tokenizer: Any, answer: str) -> int:
    """Return the single vocabulary token used to score an answer."""
    token_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if len(token_ids) != 1:
        raise ValueError(
            f"The answer {answer!r} must tokenize to exactly one token, "
            f"got {token_ids!r}."
        )
    return token_ids[0]


def build_decoder_text(
    tokenizer: Any,
    query: str,
    instruction: str,
    system_instruction: str,
    query_max_length: int,
) -> str:
    """Build the decoder-side instruction/query prompt used during training."""
    query_ids = tokenizer(
        query,
        add_special_tokens=False,
        truncation=True,
        max_length=query_max_length,
    )["input_ids"]
    truncated_query = tokenizer.decode(
        query_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return (
        "<bos><start_of_turn>user\n"
        f"{system_instruction}\n\n"
        f"<Instruct>: {instruction}\n"
        f"<Query>: {truncated_query}<end_of_turn>\n"
        "<start_of_turn>model\n\n\n\n"
    )


def get_encoder(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "get_encoder"):
        return model.get_encoder()
    if hasattr(model, "encoder"):
        return model.encoder
    raise AttributeError(f"Cannot find the encoder on {type(model).__name__}.")


def pool_encoder_chunks(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool consecutive encoder tokens while respecting padding."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    batch_size, sequence_length, hidden_size = hidden_states.shape
    num_chunks = (sequence_length + chunk_size - 1) // chunk_size
    padded_length = num_chunks * chunk_size
    pad_length = padded_length - sequence_length

    if pad_length:
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_length))
        attention_mask = F.pad(attention_mask, (0, pad_length))

    hidden_states = hidden_states.view(batch_size, num_chunks, chunk_size, hidden_size)
    chunk_mask = attention_mask.view(batch_size, num_chunks, chunk_size)
    expanded_mask = chunk_mask.unsqueeze(-1).to(hidden_states.dtype)
    pooled_hidden = (hidden_states * expanded_mask).sum(dim=2)
    pooled_hidden = pooled_hidden / chunk_mask.sum(dim=2).clamp(min=1).unsqueeze(-1)
    pooled_mask = (chunk_mask.sum(dim=2) > 0).to(attention_mask.dtype)
    return pooled_hidden, pooled_mask


def forward_reranker_model(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    decoder_attention_mask: torch.Tensor,
    encoder_chunk_size: int | None,
):
    """Run the encoder-decoder model with optional encoder token compression."""
    if encoder_chunk_size is None:
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
        )

    encoder_outputs = get_encoder(model)(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    pooled_hidden, pooled_mask = pool_encoder_chunks(
        encoder_outputs.last_hidden_state,
        attention_mask,
        encoder_chunk_size,
    )
    return model(
        encoder_outputs=BaseModelOutput(last_hidden_state=pooled_hidden),
        attention_mask=pooled_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        return_dict=True,
    )


def extract_yes_no_logits(
    logits: torch.Tensor,
    decoder_attention_mask: torch.Tensor,
    yes_token_id: int,
    no_token_id: int,
) -> torch.Tensor:
    """Extract float32 yes/no logits at each sample's final non-padding token."""
    if decoder_attention_mask.ndim != 2:
        raise ValueError("decoder_attention_mask must have shape [batch, sequence].")
    sequence_lengths = decoder_attention_mask.sum(dim=1) - 1
    if (sequence_lengths < 0).any():
        raise ValueError(
            "Every decoder input must contain at least one non-padding token."
        )

    batch_indices = torch.arange(logits.shape[0], device=logits.device)
    last_logits = logits[batch_indices, sequence_lengths]
    yes_no_logits = torch.stack(
        (last_logits[:, yes_token_id], last_logits[:, no_token_id]), dim=-1
    ).float()
    if not torch.isfinite(yes_no_logits).all():
        bad_count = (~torch.isfinite(yes_no_logits).all(dim=-1)).sum().item()
        raise RuntimeError(
            f"The model produced non-finite yes/no logits for {bad_count} input(s). "
            "Use bfloat16 or float32 instead of float16."
        )
    return yes_no_logits


def normalize_requested_dtype(dtype: Any) -> torch.dtype | None:
    """Normalize a caller-provided dtype without changing the ``auto`` behavior."""
    if dtype is None or dtype == "auto":
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    if not isinstance(dtype, str):
        return None
    normalized = dtype.lower().removeprefix("torch.")
    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(normalized)


def cast_floating_parameters(model: torch.nn.Module, dtype: torch.dtype | None) -> None:
    """Cast model parameters while preserving checkpoint buffer dtypes."""
    if dtype is None:
        return
    for parameter in model.parameters():
        if parameter.is_floating_point() and parameter.dtype != dtype:
            parameter.data = parameter.data.to(dtype=dtype)


__all__ = [
    "DEFAULT_INSTRUCTION",
    "DEFAULT_SYSTEM_INSTRUCTION",
    "answer_token_id",
    "build_decoder_text",
    "cast_floating_parameters",
    "extract_yes_no_logits",
    "forward_reranker_model",
    "get_encoder",
    "normalize_requested_dtype",
    "pool_encoder_chunks",
    "validate_text_pairs",
]
