"""HuggingFace Transformers in-process LLM adapter.

Loads any causal LM from the Hub (or a local path), exposes its full
next-token distribution. Suitable for research / experimentation; for
production-grade throughput, use a vLLM adapter (TODO).
"""

from typing import Dict, List, Optional


class TransformersClient:
    """HF Transformers in-process adapter for the LLMClient protocol.

    Holds the model and tokenizer in this process. Each call to
    next_token_logprobs runs a full forward pass on the context.

    For interactive grounding use, prefer small models (Llama 3.2 1B,
    Qwen 2.5 1.5B, GPT-2 small) on a GPU. For larger models, latency per
    token is dominated by the model forward pass, not the infinigram lookup.
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
        # Set inference mode (no dropout, no grad). Equivalent to .eval().
        self.model.train(False)

        # Stash these on self so other code can use them without re-importing torch.
        self._torch = torch
        self._vocab_size = int(self.model.config.vocab_size)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def tokenizer_id(self) -> str:
        return self.model_name

    def next_token_logprobs(self, context: List[int]) -> Dict[int, float]:
        """Run the model and return the full next-token log-probability dict.

        For a vocab of size ~100K, the returned dict has ~100K entries. This
        is fine for typical models but worth knowing for very large vocabularies.
        """
        torch = self._torch
        if not context:
            # No context: feed BOS (or 0 if there is no BOS token) so the
            # forward pass runs on something well-defined.
            bos = self.tokenizer.bos_token_id
            input_ids = torch.tensor([[bos if bos is not None else 0]], device=self.device)
        else:
            input_ids = torch.tensor([list(context)], device=self.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)

        # Materialize as a Python dict. For very large vocabularies this is
        # the expensive part of the per-token loop; future optimization could
        # keep things as tensors all the way through the mixture.
        return {i: float(lp) for i, lp in enumerate(log_probs.cpu().tolist())}
