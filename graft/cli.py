"""CLI entry point: ``graft-serve``.

Loads an LLM and an infinigram index, exposes the grounded REST API.
"""

import argparse
import sys

from graft.server.config import Config


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="graft-serve",
        description="Serve a grounded LLM (LLM + infinigram mixture) over REST.",
    )
    parser.add_argument("--config", type=str, help="YAML config file")
    parser.add_argument("--llm", type=str, help="HF model name (overrides config.llm_model)")
    parser.add_argument("--infinigram", type=str, help="Infinigram index path (overrides config.infinigram_path)")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu / mps")
    parser.add_argument("--dtype", type=str, default=None, help="float16 / bfloat16 / float32")
    args = parser.parse_args()

    if args.config:
        config = Config.from_yaml(args.config)
    else:
        if not args.llm or not args.infinigram:
            parser.error("--llm and --infinigram are required when --config is not given")
        config = Config(llm_model=args.llm, infinigram_path=args.infinigram)

    # CLI flags override config values.
    if args.llm:
        config.llm_model = args.llm
    if args.infinigram:
        config.infinigram_path = args.infinigram
    if args.port is not None:
        config.port = args.port
    if args.host is not None:
        config.host = args.host
    if args.device is not None:
        config.device = args.device
    if args.dtype is not None:
        config.torch_dtype = args.dtype

    # Imports kept here so `graft-serve --help` doesn't trigger torch / transformers loading.
    import uvicorn
    from infinigram import Infinigram
    from graft.llm.transformers import TransformersClient
    from graft.server.api import make_app

    print(f"Loading infinigram index: {config.infinigram_path}", file=sys.stderr)
    inf = Infinigram.load(config.infinigram_path)
    print(f"  {inf.n:,} tokens, vocab_size={inf.vocab_size:,}", file=sys.stderr)

    print(f"Loading LLM: {config.llm_model}", file=sys.stderr)
    llm = TransformersClient(
        config.llm_model,
        device=config.device,
        torch_dtype=config.torch_dtype,
    )
    print(f"  device={llm.device}, vocab_size={llm.vocab_size:,}", file=sys.stderr)

    if llm.vocab_size != inf.vocab_size:
        print(
            f"WARNING: vocab_size mismatch (llm={llm.vocab_size}, inf={inf.vocab_size}). "
            "Mixture will fail at the first request unless these match.",
            file=sys.stderr,
        )

    app = make_app(llm, inf, llm.tokenizer)
    print(f"Serving on http://{config.host}:{config.port}", file=sys.stderr)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
