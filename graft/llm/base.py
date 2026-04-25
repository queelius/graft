"""LLMClient protocol: the contract every LLM backend must satisfy.

The grounding pipeline only assumes a way to get next-token logprobs given a
token-id context, plus a way to identify the tokenizer family (so we can verify
alignment with the infinigram model being mixed in).
"""

from typing import Dict, List, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Contract for any LLM backend that graft can mix with an infinigram.

    All implementations must operate on token-id sequences using the same
    tokenizer as the infinigram model they're paired with. Grafted output
    is undefined if the vocabularies disagree.
    """

    @property
    def vocab_size(self) -> int:
        """Vocabulary size of the model. Should match infinigram's vocab_size."""
        ...

    def tokenizer_id(self) -> str:
        """Identifier for the tokenizer family (e.g. 'gpt2', 'meta-llama/Llama-3.2-1B').

        Used for diagnostics and for verifying alignment with the loaded
        infinigram index.
        """
        ...

    def next_token_logprobs(self, context: List[int]) -> Dict[int, float]:
        """Return next-token log probabilities given a context.

        Args:
            context: Token ids for the prompt + everything generated so far.
                     May be empty (the implementation is free to use a BOS or
                     uniform fallback, but should be deterministic).

        Returns:
            Dict mapping ``token_id -> log probability``. Implementations that
            can return the full distribution (in-process HF Transformers,
            llama.cpp, vLLM) should do so. Adapters limited to top-K (e.g.
            OpenAI API) may return only the top-K; the linear mixture handles
            missing entries by treating them as 0 in that distribution.
        """
        ...
