"""graft: ground an LLM with an infinigram corpus via mixture probability.

Usage:

    from infinigram import Infinigram
    from graft import generate_grounded, sigmoid_on_length
    from graft.llm.transformers import TransformersClient

    llm = TransformersClient("meta-llama/Llama-3.2-1B")
    inf = Infinigram.load("/path/to/index")
    tokens = generate_grounded(
        prompt=llm.tokenizer.encode("In my experience,"),
        llm=llm,
        inf=inf,
        max_tokens=50,
        alpha_fn=sigmoid_on_length(midpoint=4, max_alpha=0.6),
    )
    print(llm.tokenizer.decode(tokens))
"""

from graft.alpha import AlphaFn, constant, sigmoid_on_length, step
from graft.llm.base import LLMClient
from graft.mixture import geometric_mix, linear_mix
from graft.pipeline import generate_grounded

__version__ = "0.1.0"
__author__ = "Alexander Towell"
__email__ = "lex@metafunctor.com"

__all__ = [
    "AlphaFn",
    "LLMClient",
    "constant",
    "generate_grounded",
    "geometric_mix",
    "linear_mix",
    "sigmoid_on_length",
    "step",
]
