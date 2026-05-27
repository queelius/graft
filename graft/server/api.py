"""FastAPI REST server for graft.

OpenAI-compatible-ish ``/v1/completions`` endpoint that runs the grounded
generation pipeline. Synchronous (no streaming in v1).
"""

import time
import uuid
from functools import partial
from typing import List, Optional, Tuple, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from infinigram import Infinigram

from graft.alpha import AlphaFn, constant, sigmoid_on_length, step
from graft.llm.base import LLMClient
from graft.mixture import geometric_mix, linear_mix
from graft.pipeline import MixtureFn, generate_grounded


class CompletionRequest(BaseModel):
    """Request body for ``POST /v1/completions``."""

    prompt: Union[str, List[int]] = Field(..., description="Text or pre-tokenized ids")
    max_tokens: int = 100
    temperature: float = 1.0

    # Alpha (corpus weight) controls.
    alpha: float = Field(0.3, description="Constant alpha (used when alpha_strategy='constant')")
    alpha_strategy: str = Field("constant", description="'constant' | 'sigmoid' | 'step'")
    sigmoid_midpoint: float = 4.0
    sigmoid_steepness: float = 1.0
    sigmoid_max_alpha: float = 0.7
    step_thresholds: Optional[List[Tuple[int, float]]] = Field(
        None,
        description="(min_match_length, alpha) pairs; required when alpha_strategy='step'",
    )

    # Mixture controls.
    mixture_strategy: str = Field("linear", description="'linear' (MoE) | 'geometric' (PoE)")
    geometric_smoothing: float = Field(
        1e-8,
        description="Pseudo-count for tokens missing from a side under geometric mixture",
    )

    stop: Optional[List[str]] = None


class CompletionResponse(BaseModel):
    """Response body for ``POST /v1/completions``."""

    id: str
    created: int
    model: str
    completion: str
    tokens: List[int]
    metadata: dict


def _resolve_alpha(req: CompletionRequest) -> AlphaFn:
    if req.alpha_strategy == "constant":
        return constant(req.alpha)
    if req.alpha_strategy == "sigmoid":
        return sigmoid_on_length(
            midpoint=req.sigmoid_midpoint,
            steepness=req.sigmoid_steepness,
            max_alpha=req.sigmoid_max_alpha,
        )
    if req.alpha_strategy == "step":
        if not req.step_thresholds:
            raise HTTPException(
                status_code=400,
                detail="alpha_strategy='step' requires step_thresholds",
            )
        return step(req.step_thresholds)
    raise HTTPException(status_code=400, detail=f"Unknown alpha_strategy: {req.alpha_strategy}")


def _resolve_mixture(req: CompletionRequest) -> MixtureFn:
    if req.mixture_strategy == "linear":
        return linear_mix
    if req.mixture_strategy == "geometric":
        return partial(geometric_mix, smoothing=req.geometric_smoothing)
    raise HTTPException(
        status_code=400,
        detail=f"Unknown mixture_strategy: {req.mixture_strategy}",
    )


def make_app(llm: LLMClient, inf: Infinigram, hf_tokenizer) -> FastAPI:
    """Build the FastAPI app bound to a specific LLM/infinigram pair.

    The HF tokenizer is passed in separately so that the server can encode/decode
    text prompts and stop sequences without making the LLMClient protocol carry
    a tokenizer.
    """
    app = FastAPI(title="graft", version="0.1.0")

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "llm": llm.tokenizer_id(),
            "vocab_size": llm.vocab_size,
            "infinigram_n": inf.n,
        }

    @app.post("/v1/completions", response_model=CompletionResponse)
    async def completions(req: CompletionRequest) -> CompletionResponse:
        # Tokenize prompt.
        if isinstance(req.prompt, str):
            prompt_tokens = list(hf_tokenizer.encode(req.prompt))
        else:
            prompt_tokens = list(req.prompt)

        alpha_fn = _resolve_alpha(req)
        mixture_fn = _resolve_mixture(req)

        # Tokenize stop sequences (if any).
        stop_tokens = None
        if req.stop:
            stop_tokens = [list(hf_tokenizer.encode(s)) for s in req.stop if s]

        start = time.perf_counter()
        tokens = generate_grounded(
            prompt=prompt_tokens,
            llm=llm,
            inf=inf,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            alpha_fn=alpha_fn,
            mixture_fn=mixture_fn,
            stop_tokens=stop_tokens,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        text = hf_tokenizer.decode(tokens)

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=llm.tokenizer_id(),
            completion=text,
            tokens=tokens,
            metadata={
                "elapsed_ms": round(elapsed_ms, 2),
                "tokens_per_sec": round(len(tokens) / (elapsed_ms / 1000.0), 2) if elapsed_ms > 0 else None,
                "n_generated": len(tokens),
                "alpha_strategy": req.alpha_strategy,
                "mixture_strategy": req.mixture_strategy,
                "prompt_tokens": len(prompt_tokens),
            },
        )

    return app
