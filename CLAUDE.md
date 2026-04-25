# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`graft` mixes an LLM's next-token distribution with an
[infinigram](https://github.com/queelius/infinigram) corpus model
to "ground" the LLM in a specific corpus. Token-by-token mixture: at each
generation step, the LLM and the infinigram each propose a distribution; we
mix them via `α · p_inf + (1-α) · p_llm` and sample.

`graft` depends on `py-infinigram` and (optionally) `transformers`. The
infinigram itself is the model + your data; `graft` is the LLM-mixture-and-serve layer.

## Commands

### Testing

```bash
pytest tests/                 # all tests (no LLM/torch needed; uses FakeLLM)
pytest tests/ -v
pytest tests/test_mixture.py  # single file
```

### Development

```bash
pip install -e .[dev]             # core + tests
pip install -e .[transformers,dev]  # add HF Transformers adapter
graft-serve --llm gpt2 --infinigram /path/to/index --port 8000
```

## Architecture

### Layers

```
HTTP API (OpenAI-compatible-ish)            graft/server/api.py
       └─ generate_grounded(...) per request
Pipeline (token-by-token mixture loop)      graft/pipeline.py
       ├─ LLMClient.next_token_logprobs(context) → log-probs
       ├─ Infinigram.continuations(context) → counts (or None)
       ├─ alpha_fn(match_length) → α
       ├─ mixture_fn(p_llm, p_inf, α) → mixed distribution
       └─ sample_from_distribution(p, temperature) → next token
LLM clients                                  graft/llm/{base,transformers}.py
Mixture / alpha primitives                   graft/{mixture,alpha}.py
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `mixture.py` | Pure functions: `linear_mix` (MoE), `geometric_mix` (PoE) |
| `alpha.py` | α strategies: `constant`, `sigmoid_on_length`, `step` |
| `llm/base.py` | `LLMClient` Protocol; minimal contract for any backend |
| `llm/transformers.py` | HF Transformers in-process adapter |
| `pipeline.py` | `generate_grounded` token loop |
| `server/api.py` | FastAPI app factory, `/v1/completions`, `/health` |
| `server/config.py` | YAML config schema (Pydantic) |
| `cli.py` | `graft-serve` entry point |

### Critical contract: tokenizer alignment

`graft` mixes distributions over a **shared vocabulary**. The LLM and the
infinigram MUST use the same tokenizer. `generate_grounded` raises
`ValueError` if `llm.vocab_size != inf.vocab_size`.

This means: pick the LLM first, then build the infinigram with its tokenizer.

### LLMClient protocol

Anything implementing this works as an LLM backend:

```python
class LLMClient(Protocol):
    @property
    def vocab_size(self) -> int: ...
    def tokenizer_id(self) -> str: ...
    def next_token_logprobs(self, context: List[int]) -> Dict[int, float]: ...
```

Adapters that can return the full distribution (Transformers, vLLM,
llama.cpp) get true mixture math. Adapters limited to top-K (OpenAI API)
return a partial dict; `linear_mix` handles missing entries by treating
them as zero in that distribution.

## Mixture math: design choice

Two strategies live in `graft.mixture`:

- **Linear (default)**: `α · p_inf + (1-α) · p_llm`. Mixture of Experts.
  Preserves LLM coverage: tokens outside infinigram's support survive with
  weight `(1-α) · p_llm`.
- **Geometric**: `p_inf^α · p_llm^(1-α) / Z`. Product of Experts.
  Aggressive (only tokens both endorse survive). Requires smoothing on
  sparse infinigram or it collapses to the support intersection.

For grounding (which is the project's purpose), linear is the right default.
Geometric exists for cases where you want agreement-only / terminology
enforcement / distillation-style behavior.

## Test strategy

Tests use a `FakeLLM` (`tests/test_pipeline.py`) so the test suite is
CPU-only and fast. No HF model loading in tests. The HF Transformers
adapter is exercised manually / via `graft-serve` smoke testing during
development.

## Performance notes

- Per-token cost is dominated by the LLM forward pass (tens of ms on GPU
  for small models). Infinigram lookup is sub-ms after the binary-search
  optimization in v0.7.
- The full LLM distribution is materialized as a `Dict[int, float]` of
  size `vocab_size` per step. For ~100K vocab this is fine; for larger
  vocabs, consider a future optimization to keep tensors throughout the
  mixture.
