# graft

Ground an LLM in a corpus by mixing the LLM's next-token distribution with an
[infinigram](https://github.com/queelius/infinigram) corpus model, per token.

The mixture is computed as `P_final = α · P_infinigram + (1-α) · P_LLM`,
where α can be a constant or a function of the longest matching corpus suffix.
Long matches → trust the corpus; short or no matches → fall back to the LLM.

Build an infinigram on your data once. Plug in any HF causal LM. Get an
LLM that's grounded in your corpus, served via a REST API.

## Status

Alpha. v0.1 is a working synchronous server with one LLM adapter (HF Transformers).
Future work: streaming, vLLM adapter, llama.cpp adapter, OpenAI API adapter
(approximate mixture from top-K logprobs).

## Install

Editable install for development:

```bash
git clone https://github.com/queelius/graft
cd graft
pip install -e .[transformers,dev]
```

You also need `infinigram` installed (sibling project, also editable):

```bash
pip install -e ../infinigram
```

## Quick start (Python)

```python
from infinigram import Infinigram
from graft import generate_grounded, sigmoid_on_length
from graft.llm.transformers import TransformersClient

llm = TransformersClient("gpt2")
inf = Infinigram.load("/path/to/your/index")  # built with the SAME tokenizer

# Same tokenizer family on both sides!
prompt = llm.tokenizer.encode("In my experience,")

tokens = generate_grounded(
    prompt=prompt,
    llm=llm,
    inf=inf,
    max_tokens=50,
    temperature=0.7,
    alpha_fn=sigmoid_on_length(midpoint=4, max_alpha=0.6),
)
print(llm.tokenizer.decode(tokens))
```

## Quick start (REST)

```bash
graft-serve --llm gpt2 --infinigram /path/to/index --port 8000
```

Then in another terminal:

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "In my experience,",
    "max_tokens": 50,
    "temperature": 0.7,
    "alpha_strategy": "sigmoid"
  }'
```

## How α works

α is the weight on the corpus distribution in the mixture. Strategies:

- **`constant(value)`**: always-the-same α. Useful as a baseline.
- **`sigmoid_on_length(midpoint, steepness, max_alpha)`**: α grows with the
  longest matching corpus suffix. Short matches → α near 0; long matches →
  α near `max_alpha`. Sensible default for "trust the corpus when it has
  evidence."
- **`step(thresholds)`**: discrete confidence regimes. e.g.
  `step([(0, 0.0), (3, 0.3), (6, 0.6)])` means "0 below 3 tokens, 0.3 from
  3-5, 0.6 from 6+."

## Tokenizer alignment

The mixture only makes sense if the LLM and infinigram use the **same
tokenizer**. Build your infinigram with the LLM's tokenizer:

```python
from infinigram import Infinigram
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
text = open("my_corpus.txt").read()
Infinigram.build(tok.encode(text), "my_index", tokenizer=tok)
```

Mixing across tokenizers (e.g. cl100k infinigram + Llama LLM) is undefined
behavior; `graft` will refuse with a `ValueError` at generation time.

## Mixture strategies

Two implementations live in `graft.mixture`:

- **`linear_mix`** (default): `α · p_inf + (1-α) · p_llm`. The standard
  Mixture-of-Experts shape. Tokens missing from one side fall through.
- **`geometric_mix`**: `p_inf^α · p_llm^(1-α) / Z` (Product of Experts).
  Sharper, requires smoothing for sparse infinigram. Use when you want
  agreement-only behavior.

Most users should stick with linear.

## License

MIT.
