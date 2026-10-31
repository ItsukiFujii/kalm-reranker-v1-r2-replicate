"""Offline Cog entrypoint for the KaLM-Reranker-V1-R2 models."""

from copy import copy
import os
from pathlib import Path

from cog import BaseRunner, Input
from typing_extensions import NotRequired, TypedDict


class RerankResult(TypedDict):
    index: int
    relevance_score: float
    document: NotRequired[str]


class Runner(BaseRunner):
    def setup(self) -> None:
        """Load the bundled model once and reuse its weights across requests."""
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        import torch

        from weights.kalm_reranker import KaLMReranker

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = (
            torch.bfloat16
            if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        self.reranker = KaLMReranker(
            str(Path(__file__).resolve().parent / "weights"),
            device=device,
            dtype=dtype,
            batch_size=8 if device.type == "cuda" else 1,
            query_max_length=512,
            max_length=1024,
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="sdpa",
        )
        self.reranker.model.requires_grad_(False)
        self.reranker.model.config.decoder.use_cache = False

    def run(
        self,
        query: str = Input(description="Query text.", min_length=1),
        documents: list[str] = Input(description="Array of candidate document texts."),
        compression_ratio: int = Input(
            default=4,
            ge=1,
            le=128,
            description=(
                "Document representation compression factor, matching the official "
                "chunk_size parameter. Mean-pool this many consecutive encoder "
                "tokens into one representation. Use 1 for no compression. "
                "Supported range: integer factors from 1 to 128."
            ),
        ),
        top_n: int | None = Input(
            default=None,
            ge=0,
            description=(
                "Maximum number of ranked results. Omit to return all documents; "
                "0 returns no results. Values above the document count return all."
            ),
        ),
        return_documents: bool = Input(
            default=False,
            description="Include the original document text in each result.",
        ),
        instruction: str = Input(
            default="Given a query, retrieve documents that answer the query.",
            description="Task instruction inserted into the official decoder prompt.",
        ),
    ) -> list[RerankResult]:
        """Return descending P(yes) scores; ties preserve original document order."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if not isinstance(documents, list):
            raise ValueError("documents must be an array of strings.")
        for index, document in enumerate(documents):
            if not isinstance(document, str):
                raise ValueError(f"documents[{index}] must be a string.")
        if type(compression_ratio) is not int or not 1 <= compression_ratio <= 128:
            raise ValueError("compression_ratio must be an integer from 1 to 128.")
        if top_n is not None and (type(top_n) is not int or top_n < 0):
            raise ValueError("top_n must be a non-negative integer or null.")
        if type(return_documents) is not bool:
            raise ValueError("return_documents must be a boolean.")
        if not isinstance(instruction, str):
            raise ValueError("instruction must be a string.")
        if not documents or top_n == 0:
            return []

        # Only the lightweight wrapper is copied. Model weights and the tokenizer
        # remain shared, while the request's compression factor is independent.
        # The official forward path passes this chunk_size to encoder mean pooling
        # and supplies the compressed states/mask to the scoring decoder.
        # No document representations or scores are cached between requests.
        request_reranker = copy(self.reranker)
        request_reranker.chunk_size = compression_ratio
        ranking = request_reranker.rank(
            query,
            documents,
            instruction=instruction,
            top_k=top_n,
        )
        results: list[RerankResult] = []
        for item in ranking:
            index = int(item["corpus_id"])
            result: RerankResult = {
                "index": index,
                "relevance_score": float(item["score"]),
            }
            if return_documents:
                result["document"] = documents[index]
            results.append(result)
        return results
