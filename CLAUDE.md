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
pytest tests/ -k linear_mix   # single test by keyword
pytest tests/ --cov=graft     # coverage (pytest-cov is in [dev])
```

### Development

```bash
pip install -e .[dev]             # core + tests
pip install -e .[transformers,dev]  # add HF Transformers adapter
graft-serve --llm gpt2 --infinigram /path/to/index --port 8000
graft-serve --config server.yaml  # alternative: YAML config (see server/config.py)
```

## End-to-end: build, serve, query

The tokenizer flows top-down. Pick the LLM first; every downstream step
inherits its tokenizer. Mismatch at any point is caught either at server
boot (warning) or at the first generation (`ValueError`).

```python
# 1. Build the infinigram once, using the LLM's tokenizer
from transformers import AutoTokenizer
from infinigram import Infinigram
tok = AutoTokenizer.from_pretrained("gpt2")
ids = tok.encode(open("my_corpus.txt").read())
Infinigram.build(ids, "my_index/", tokenizer=tok)
```

```bash
# 2. Serve the (LLM, infinigram) pair (both stay loaded for the process life)
graft-serve --llm gpt2 --infinigram my_index/ --port 8000

# 3. Query
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "In my experience,", "max_tokens": 50, "alpha_strategy": "sigmoid"}'
```

Lifecycle checkpoints:
- **Step 1** writes the index to disk; this is the only step that is
  expensive per-corpus and not per-request.
- **Step 2** loads both, prints `vocab_size` for each, and **warns but does
  not abort** on mismatch (`cli.py:65-71`).
- **Step 3**'s first request triggers the **hard check**
  (`pipeline.py:62-66`), which raises `ValueError` on mismatch.

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

### Pipeline early exits

`generate_grounded` has two fast paths that bypass mixture entirely
(`pipeline.py`):

- `inf.continuations(context) is None` → no suffix match in corpus, return
  `p_llm` directly. `alpha_fn` is **not called**.
- `alpha_fn(match_length) <= 0.0` → mixture skipped, return `p_llm` directly.
  `mixture_fn` is **not called**.

Stop tokens are **stripped** from the returned token list when matched (the
match suffix is trimmed). Callers comparing expected vs. actual should not
expect the stop sequence in the output.

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

## Server contract

- `make_app(llm, inf, hf_tokenizer)` (`server/api.py`) closes over a single
  LLM + infinigram pair for the process lifetime. No per-request model
  selection.
- The REST API exposes the full pipeline surface:
  - `alpha_strategy`: `"constant"` | `"sigmoid"` | `"step"`. For `"step"`,
    pass `step_thresholds` as a list of `[match_length, alpha]` pairs.
  - `mixture_strategy`: `"linear"` (default, MoE) | `"geometric"` (PoE).
    `geometric_smoothing` is a per-request pseudo-count for missing tokens.
  - Resolution lives in `_resolve_alpha` and `_resolve_mixture` in
    `server/api.py`; both are unit-tested in `tests/test_server.py`.
- Heavy imports (`torch`, `transformers`, `uvicorn`) are deferred to
  `__init__` / `main()` bodies, so `graft-serve --help` and importing
  `graft.llm.transformers` do not require torch.

## Test strategy

Two tiers, run together by default:

- **Pure / fake-LLM tests** (`test_alpha.py`, `test_mixture.py`,
  `test_pipeline.py`, `test_server.py`, `test_cache_split.py`) use a
  byte-vocab `FakeLLM` and exercise the full pipeline + REST surface
  without loading any HF model. Sub-second.
- **Real-model integration tests** (`test_transformers_kv_cache.py`,
  `test_end_to_end.py`) load `sshleifer/tiny-gpt2` (a ~hundred-KB toy
  model used in HF's own test fixtures) and build a real `Infinigram`
  index over a few sentences. They guard the things only a real stack
  can verify: KV-cache numerical equivalence, tokenizer alignment, and
  the alpha=0 / alpha=1 limits of `generate_grounded` reducing
  respectively to bare-LLM argmax and corpus-driven continuation.
  Each takes a few seconds; all `pytest.importorskip` torch /
  transformers so the tier auto-disables on a `[dev]`-only install.

## Performance notes

- Per-token cost is dominated by the LLM forward pass (tens of ms on GPU
  for small models). Infinigram lookup is sub-ms after the binary-search
  optimization in v0.7.
- **KV-cache reuse**: `TransformersClient` caches `past_key_values` and
  detects strict-prefix extensions of the previous context (the common
  case during a single generation). This collapses an O(n²) sequence of
  forward passes to one prefill plus n single-token forwards. Logic is in
  `_cache_split` (`llm/transformers.py`); cache-correctness is verified
  by `tests/test_transformers_kv_cache.py`, which compares cached vs
  reset-recomputed logits against `sshleifer/tiny-gpt2`. The cache is
  guarded by a `threading.Lock` so concurrent FastAPI requests serialize
  through the model rather than corrupting cache state.
- The full LLM distribution is materialized as a `Dict[int, float]` of
  size `vocab_size` per step. For ~100K vocab this is fine; for larger
  vocabs, consider a future optimization to keep tensors throughout the
  mixture.

## Future work / known gaps

Listed here so future sessions treat these as design boundaries, not bugs:

- **No streaming**: `/v1/completions` returns after the full generation
  completes. SSE would require yielding inside `generate_grounded` and a
  server adapter.
- **Single LLM adapter**: only HF Transformers in-process. vLLM,
  llama.cpp, and OpenAI top-K adapters are TODO; `LLMClient` was designed
  for them, and `linear_mix` already handles partial dicts by treating
  missing entries as zero.
- **No batching, no per-request model swap**: the server holds one
  `(llm, inf)` pair and serves requests serially through the same model.
- **Dict-based distributions**: `p_llm` and `p_inf` are
  `Dict[int, float]` of size `vocab_size` per step. For >100K vocabs, a
  tensor-native mixture pipeline would avoid the per-step Python overhead.
