"""Alpha strategies: how much to trust infinigram given match length.

Each strategy is a *factory* that returns a callable ``alpha_fn(match_length: int) -> float``.
Pass the returned callable to :func:`graft.pipeline.generate_grounded`.

Conceptually: ``alpha = 0`` means "ignore infinigram, use LLM only";
``alpha = 1`` means "trust infinigram fully, ignore LLM."
Match length is the number of tokens of the current context that exist as a
contiguous substring of the corpus. Longer = more confident grounding signal.
"""

import math
from typing import Callable, Iterable, Tuple

AlphaFn = Callable[[int], float]


def constant(value: float) -> AlphaFn:
    """Always return the same alpha regardless of match length.

    Useful as a baseline. ``constant(0.0)`` disables grounding; ``constant(1.0)``
    means corpus-only generation; ``constant(0.3)`` is a reasonable starting mix.
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {value}")

    def _fn(match_length: int) -> float:
        return value

    return _fn


def sigmoid_on_length(
    midpoint: float = 4.0,
    steepness: float = 1.0,
    max_alpha: float = 0.7,
) -> AlphaFn:
    """Sigmoid that grows with match length, asymptotic at ``max_alpha``.

    At ``match_length == midpoint`` the result is ``max_alpha / 2``.
    At ``match_length == 0`` the result is exactly 0 (no signal at all).

    Args:
        midpoint: match length at which alpha hits half its ceiling.
        steepness: divisor in the sigmoid; larger = smoother transition.
        max_alpha: ceiling for alpha. Keep < 1.0 to retain some LLM contribution.
    """
    if steepness <= 0:
        raise ValueError("steepness must be positive")
    if not 0.0 <= max_alpha <= 1.0:
        raise ValueError("max_alpha must be in [0, 1]")

    def _fn(match_length: int) -> float:
        if match_length <= 0:
            return 0.0
        x = (match_length - midpoint) / steepness
        return max_alpha / (1.0 + math.exp(-x))

    return _fn


def step(thresholds: Iterable[Tuple[int, float]]) -> AlphaFn:
    """Step function: map match length to alpha via thresholds.

    ``thresholds`` is an iterable of ``(min_match_length, alpha)`` pairs.
    For each match length, the alpha of the largest threshold <= match length
    is returned. Below all thresholds, alpha is 0.

    Example:
        >>> fn = step([(0, 0.0), (3, 0.3), (6, 0.5), (10, 0.7)])
        >>> fn(2)   # 0.0
        >>> fn(5)   # 0.3
        >>> fn(7)   # 0.5
        >>> fn(20)  # 0.7
    """
    sorted_thresholds = sorted(thresholds)
    if not sorted_thresholds:
        raise ValueError("thresholds must not be empty")

    def _fn(match_length: int) -> float:
        chosen = 0.0
        for threshold, alpha in sorted_thresholds:
            if match_length >= threshold:
                chosen = alpha
        return chosen

    return _fn
