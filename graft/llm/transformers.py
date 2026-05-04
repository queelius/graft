"""HuggingFace Transformers in-process LLM adapter.

Loads any causal LM from the Hub (or a local path), exposes its full
next-token distribution. Suitable for research / experimentation; for
production-grade throughput, use a vLLM adapter (TODO).

Reuses the model's KV cache between calls when the new context strictly
extends the previous one (the common case during a single generation).
This collapses an O(n^2) sequence of forward passes to one prefill plus
n single-token forwards.
"""

import threading
from typing import Dict, List, Optional, Tuple


def _cache_split(
    context: List[int],
    cached_context: Optional[List[int]],
) -> Tuple[bool, List[int]]:
    """Decide whether to reuse the KV cache or do a full re-forward.

    Returns ``(extends_cache, tokens_to_forward)``.

    - ``extends_cache=True``: ``cached_context`` is a strict prefix of
      ``context``; only the new suffix needs to be forwarded.
    - ``extends_cache=False``: cache miss; ``tokens_to_forward`` is the
      full ``context`` (callers should discard the cache).

    Equal contexts return ``(False, context)`` because forwarding 0 new
    tokens against the existing cache cannot produce a logits row at the
    new position.
    """
    if (
        cached_context
        and len(context) > len(cached_context)
        and context[: len(cached_context)] == cached_context
    ):
        return True, context[len(cached_context):]
    return False, context


class TransformersClient:
    """HF Transformers in-process adapter for the LLMClient protocol.

    Holds the model, tokenizer, and a KV cache in this process. Forward
    passes are serialized through an internal lock so concurrent requests
    on a long-lived server (e.g. via FastAPI's threadpool) cannot corrupt
    cache state.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
    ):
        """Load a HF causal LM and its tokenizer.

        Args:
            model_name: HF Hub repo id or local path (e.g.
                'meta-llama/Llama-3.2-1B', 'gpt2').
            device: 'cuda', 'cpu', 'mps', or None to autodetect.
            torch_dtype: 'float16', 'bfloat16', 'float32', or None for default.
        """
        # Imports kept inside __init__ so that listing the protocol via
        # `from graft.llm.transformers import TransformersClient` doesn't
        # force-import torch unless the user actually instantiates one.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        load_kwargs = {}
        if torch_dtype:
            load_kwargs["torch_dtype"] = getattr(torch, torch_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.to(self.device)
        # Inference mode: no dropout, no grad accumulation.
        self.model.train(False)

        # Stash these on self so other code can use them without re-importing torch.
        self._torch = torch
        self._vocab_size = int(self.model.config.vocab_size)

        # KV-cache state, guarded by _lock.
        self._cached_context: Optional[List[int]] = None
        self._past_key_values = None
        self._lock = threading.Lock()

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def tokenizer_id(self) -> str:
        return self.model_name

    def reset_cache(self) -> None:
        """Discard the KV cache.

        Call between unrelated generation contexts. Not required for
        correctness (divergent contexts auto-trigger a cache miss) but
        can free GPU memory between large generations.
        """
        with self._lock:
            self._cached_context = None
            self._past_key_values = None

    def next_token_logprobs(self, context: List[int]) -> Dict[int, float]:
        """Run the model and return the full next-token log-probability dict.

        For a vocab of size ~100K, the returned dict has ~100K entries. This
        is fine for typical models but worth knowing for very large vocabularies.
        """
        torch = self._torch

        # Resolve effective context (BOS / 0 for empty).
        if not context:
            bos = self.tokenizer.bos_token_id
            effective_context = [bos if bos is not None else 0]
        else:
            effective_context = list(context)

        with self._lock:
            extends, to_forward = _cache_split(effective_context, self._cached_context)
            past = self._past_key_values if extends else None

            input_ids = torch.tensor([to_forward], device=self.device)
            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    past_key_values=past,
                    use_cache=True,
                )
                logits = outputs.logits[0, -1, :]
                log_probs = torch.log_softmax(logits, dim=-1)

            self._cached_context = effective_context
            self._past_key_values = outputs.past_key_values

            result = {i: float(lp) for i, lp in enumerate(log_probs.cpu().tolist())}

        return result
