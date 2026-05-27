"""Shared test helpers.

Lives in conftest so both ``test_pipeline.py`` and ``test_server.py``
can import the same ``FakeLLM`` without duplicating it.
"""

import math
from typing import Dict, List


class FakeLLM:
    """Deterministic LLM for tests: returns a configurable distribution.

    Vocab size matches infinigram's byte vocab (256) by default so the
    alignment check passes.
    """

    def __init__(self, vocab_size: int = 256, distribution: Dict[int, float] = None):
        self._vocab_size = vocab_size
        if distribution is None:
            distribution = {i: 1.0 / vocab_size for i in range(vocab_size)}
        z = sum(distribution.values())
        self._logprobs = {v: math.log(p / z) for v, p in distribution.items() if p > 0}

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def tokenizer_id(self) -> str:
        return "fake"

    def next_token_logprobs(self, context: List[int]) -> Dict[int, float]:
        return dict(self._logprobs)
