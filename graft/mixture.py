"""Mixture functions for combining LLM and infinigram distributions.

Two strategies, both pluggable as ``mixture_fn`` in :func:`graft.pipeline.generate_grounded`.

- :func:`linear_mix` (Mixture of Experts): ``alpha * p_inf + (1-alpha) * p_llm``
  Tokens outside infinigram's support fall through to LLM-only weight.
  Default and recommended for grounding.

- :func:`geometric_mix` (Product of Experts): ``p_inf^alpha * p_llm^(1-alpha) / Z``
  Requires nonzero support; tokens missing in either side need smoothing.
  More aggressive (only tokens both endorse survive). Useful for terminology
  enforcement or distillation-style behavior.
"""

from typing import Dict


def linear_mix(
    p_llm: Dict[int, float],
    p_inf: Dict[int, float],
    alpha: float,
) -> Dict[int, float]:
    """Linear mixture of two distributions.

    ``p_final(v) = alpha * p_inf(v) + (1 - alpha) * p_llm(v)``

    Tokens missing from either side are treated as zero in that distribution,
    so the LLM's broad support is preserved (any token in the LLM's distribution
    survives with weight at least ``(1-alpha) * p_llm(v)``). The result is a
    valid distribution if both inputs are.
    """
    keys = p_llm.keys() | p_inf.keys()
    return {
        v: alpha * p_inf.get(v, 0.0) + (1.0 - alpha) * p_llm.get(v, 0.0)
        for v in keys
    }


def geometric_mix(
    p_llm: Dict[int, float],
    p_inf: Dict[int, float],
    alpha: float,
    smoothing: float = 1e-8,
) -> Dict[int, float]:
    """Geometric mixture (Product of Experts) over the union of supports.

    ``p_final(v) ~ p_inf(v)^alpha * p_llm(v)^(1-alpha)``, then renormalized.

    Tokens missing from either distribution are assigned ``smoothing`` so the
    product is well-defined. Without smoothing, tokens outside the intersection
    of supports get zero mass and the mixture collapses to whatever both
    distributions endorse.

    Note: with sparse infinigram, geometric mixture severely restricts the
    output vocabulary unless ``smoothing`` is meaningful. Prefer
    :func:`linear_mix` for grounding unless you specifically want the
    "agreement-only" behavior.
    """
    keys = p_llm.keys() | p_inf.keys()
    raw = {
        v: (p_inf.get(v, smoothing) ** alpha)
           * (p_llm.get(v, smoothing) ** (1.0 - alpha))
        for v in keys
    }
    z = sum(raw.values())
    if z <= 0.0:
        # Fully degenerate. Fall back to uniform over the union.
        return {v: 1.0 / len(keys) for v in keys}
    return {v: r / z for v, r in raw.items()}
