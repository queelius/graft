"""Token-by-token generation loop that mixes LLM and infinigram per step."""

import math
from typing import Callable, Dict, List, Optional

from infinigram import Infinigram
from infinigram.sampling import sample_from_distribution

from graft.alpha import AlphaFn, constant
from graft.llm.base import LLMClient
from graft.mixture import linear_mix


MixtureFn = Callable[[Dict[int, float], Dict[int, float], float], Dict[int, float]]


def generate_grounded(
    prompt: List[int],
    llm: LLMClient,
    inf: Infinigram,
    max_tokens: int = 100,
    temperature: float = 1.0,
    alpha_fn: Optional[AlphaFn] = None,
    mixture_fn: Optional[MixtureFn] = None,
    stop_tokens: Optional[List[List[int]]] = None,
) -> List[int]:
    """Generate tokens by mixing LLM and infinigram next-token distributions.

    Per step:
      1. Get next-token log-probs from the LLM (full distribution if available).
      2. Get continuation counts from the infinigram for the longest matching suffix.
      3. Compute alpha from the match length via ``alpha_fn``.
      4. Mix the two distributions via ``mixture_fn``.
      5. Sample (with temperature).
      6. Append to context, check stop conditions.

    When the infinigram returns no continuations (``inf.continuations(...) is None``),
    the mixture degenerates to LLM-only for that step (alpha = 0).

    Args:
        prompt: Starting context as token ids. Must use the same tokenizer as
            ``inf`` and ``llm``.
        llm: LLMClient.
        inf: Infinigram model.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature; 0 means greedy (argmax).
        alpha_fn: ``match_length -> alpha`` callable. Default: ``constant(0.3)``.
        mixture_fn: ``(p_llm, p_inf, alpha) -> p_mixed`` callable.
            Default: :func:`graft.mixture.linear_mix`.
        stop_tokens: Optional stop sequences (lists of token ids). Generation
            halts and these tokens are stripped if any sequence matches the tail
            of generated tokens.

    Returns:
        Generated token ids (not including the prompt).
    """
    if alpha_fn is None:
        alpha_fn = constant(0.3)
    if mixture_fn is None:
        mixture_fn = linear_mix

    if llm.vocab_size != inf.vocab_size:
        raise ValueError(
            f"Tokenizer mismatch: LLM vocab_size={llm.vocab_size}, "
            f"infinigram vocab_size={inf.vocab_size}. Both must use the same tokenizer."
        )

    context = list(prompt)
    generated: List[int] = []

    for _ in range(max_tokens):
        # 1. LLM step.
        log_probs = llm.next_token_logprobs(context)
        p_llm = {v: math.exp(lp) for v, lp in log_probs.items()}

        # 2. Infinigram step.
        inf_counts = inf.continuations(context)
        if inf_counts is None:
            # No suffix match → no signal; LLM-only.
            p_final = p_llm
        else:
            _, match_length = inf.longest_suffix(context)
            alpha = alpha_fn(match_length)
            if alpha <= 0.0:
                p_final = p_llm
            else:
                total = sum(inf_counts.values())
                p_inf = {v: c / total for v, c in inf_counts.items()}
                # 3-4. Mix.
                p_final = mixture_fn(p_llm, p_inf, alpha)

        # 5. Sample.
        next_token = sample_from_distribution(p_final, temperature)
        generated.append(next_token)
        context.append(next_token)

        # 6. Stop check.
        if stop_tokens:
            for stop_seq in stop_tokens:
                k = len(stop_seq)
                if k > 0 and len(generated) >= k and generated[-k:] == list(stop_seq):
                    return generated[:-k]

    return generated
