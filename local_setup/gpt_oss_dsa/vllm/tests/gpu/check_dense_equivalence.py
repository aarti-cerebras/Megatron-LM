"""Token-exact control: with `index_topk` at least every sequence length, every query's top-k is
every legal key, and the DSA model must reproduce stock GPT-OSS greedy output exactly. This
exercises the whole serving path (weights, sinks, GQA grouping, block table, indexer kernels)
without needing a checkpoint that trained sparse, so a Phase-1 serving dir built with
`--index-topk 4096` is the right input.

Each model runs in its own subprocess because a vLLM engine does not release its GPU cleanly
in-process. Both run greedy, eager, block size 64, prefix caching off.

  uv run tests/gpu/check_dense_equivalence.py --serving-dir <dir built with --index-topk 4096> \\
      --baseline /cb/ml-eng/aarti/models/gpt-oss-20b --max-tokens 64
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "In 1905, Albert Einstein published four papers that",
    "Q: What is 17 * 23?\nA:",
    "Translate to German: The weather is nice today, so we will go hiking in the mountains.",
]


def run_model(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    prompts = json.loads(Path(args.prompts_json).read_text())
    extra = {}
    if args.attention_backend:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        extra["attention_backend"] = AttentionBackendEnum[args.attention_backend]
    llm = LLM(
        model=args.model,
        **extra,
        tensor_parallel_size=args.tp,
        block_size=64,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        # Required by the DSA model (gpt_oss_dsa/model.py). Stock GPT-OSS cannot take it on
        # vLLM 0.26.0: its sliding-window blocks are sized 16 against 64 for full attention and
        # the disabled-hybrid conversion then fails, so the baseline runs with vLLM's defaults.
        disable_hybrid_kv_cache_manager=args.disable_hybrid_kv_cache_manager,
        dtype="bfloat16",
        seed=0,
    )
    # Two logprobs per step: the chosen token and the runner-up. Their gap at a divergence says
    # whether a mismatch is a near-tie flipped by summation order or a real deviation.
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=2)
    outputs = llm.generate(prompts, params)
    result = []
    for o in outputs:
        completion = o.outputs[0]
        top2 = []
        for step in completion.logprobs or []:
            ranked = sorted((lp.logprob for lp in step.values()), reverse=True)
            top2.append(ranked[:2] if len(ranked) >= 2 else ranked + [float("-inf")])
        result.append(
            {
                "prompt": o.prompt,
                "token_ids": list(completion.token_ids),
                "text": completion.text,
                "top2_logprobs": top2,
            }
        )
    Path(args.result_json).write_text(json.dumps(result, indent=1))


def spawn(
    model: str,
    prompts_json: str,
    result_json: str,
    args: argparse.Namespace,
    env: dict,
    disable_hybrid_kv_cache_manager: bool,
    attention_backend: str | None = None,
) -> None:
    cmd = [
        sys.executable, __file__, "--run",
        "--model", model,
        "--prompts-json", prompts_json,
        "--result-json", result_json,
        "--max-tokens", str(args.max_tokens),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--tp", str(args.tp),
    ]
    if disable_hybrid_kv_cache_manager:
        cmd.append("--disable-hybrid-kv-cache-manager")
    if attention_backend:
        cmd += ["--attention-backend", attention_backend]
    print("[dense-equivalence]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env={**os.environ, **env})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serving-dir")
    ap.add_argument("--baseline", help="stock openai/gpt-oss-20b directory")
    ap.add_argument("--prompts", help="file with one prompt per line; default: built-in list")
    ap.add_argument("--long-prompt-words", type=int, default=0,
                    help="also include a synthetic prompt of about this many words, so the selection "
                         "spans many KV blocks (block size 64) instead of one or two")
    ap.add_argument("--noise-floor-backend", default=None,
                    help="also run the stock model on this vLLM attention backend (e.g. TRITON_ATTN) and "
                         "report stock-vs-stock differences, the noise floor of dense kernel choice")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    ap.add_argument("--tp", type=int, default=1)
    # Internal, one model per subprocess.
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--prompts-json")
    ap.add_argument("--result-json")
    ap.add_argument("--disable-hybrid-kv-cache-manager", action="store_true")
    ap.add_argument("--attention-backend", default=None, help="internal: dense backend for a --run")
    args = ap.parse_args()

    if args.run:
        run_model(args)
        return 0

    if not (args.serving_dir and args.baseline):
        ap.error("--serving-dir and --baseline are required")
    config = json.loads((Path(args.serving_dir) / "config.json").read_text())
    if config["index_topk"] < args.max_model_len:
        print(
            f"warning: index_topk={config['index_topk']} < max_model_len={args.max_model_len}; "
            "prompts longer than index_topk minus max_tokens are not dense-equivalent",
            flush=True,
        )
    prompts = (
        [line.rstrip("\n") for line in Path(args.prompts).read_text().splitlines() if line.strip()]
        if args.prompts
        else list(DEFAULT_PROMPTS)
    )
    if args.long_prompt_words:
        filler = (
            "The expedition kept a daily log of weather, supplies and distance covered. "
            "Each entry was numbered and signed by the officer on watch. "
        )
        repeats = max(1, args.long_prompt_words // len(filler.split()))
        prompts.append(filler * repeats + "Summarize the log-keeping procedure in one sentence:")

    with tempfile.TemporaryDirectory(prefix="dsa_dense_eq_") as tmp:
        prompts_json = f"{tmp}/prompts.json"
        Path(prompts_json).write_text(json.dumps(prompts))
        baseline_json = f"{tmp}/baseline.json"
        dsa_json = f"{tmp}/dsa.json"
        spawn(args.baseline, prompts_json, baseline_json, args, {}, disable_hybrid_kv_cache_manager=False)
        spawn(
            args.serving_dir,
            prompts_json,
            dsa_json,
            args,
            {"GPT_OSS_DSA_ALLOW_PHASE1": "1", "GPT_OSS_DSA_CHECK_SELECTION": "1"},
            disable_hybrid_kv_cache_manager=True,
        )
        baseline = json.loads(Path(baseline_json).read_text())
        dsa = json.loads(Path(dsa_json).read_text())
        alt = None
        if args.noise_floor_backend:
            alt_json = f"{tmp}/baseline_{args.noise_floor_backend}.json"
            spawn(
                args.baseline,
                prompts_json,
                alt_json,
                args,
                {},
                disable_hybrid_kv_cache_manager=False,
                attention_backend=args.noise_floor_backend,
            )
            alt = json.loads(Path(alt_json).read_text())

    if alt is not None:
        print(f"--- stock FA3 vs stock {args.noise_floor_backend} (dense kernel noise floor) ---")
        report(baseline, alt, "alt")
        print("--- stock FA3 vs DSA ---")
    return report(baseline, dsa, "dsa")


def report(baseline: list[dict], other: list[dict], label: str) -> int:
    mismatches = 0
    for b, d in zip(baseline, other):
        first = next(
            (i for i, (x, y) in enumerate(zip(b["token_ids"], d["token_ids"])) if x != y),
            min(len(b["token_ids"]), len(d["token_ids"])),
        )
        # Over the shared prefix both models chose the same token; the largest difference in its
        # logprob is the numerical noise floor between the dense and the sparse attend.
        noise = max(
            (abs(x[0] - y[0]) for x, y in zip(b["top2_logprobs"][:first], d["top2_logprobs"][:first])),
            default=0.0,
        )
        if b["token_ids"] == d["token_ids"]:
            print(f"MATCH   {len(b['token_ids']):3d} tokens  noise {noise:.2e}  {b['prompt'][:50]!r}")
            continue
        mismatches += 1
        gap_stock = b["top2_logprobs"][first][0] - b["top2_logprobs"][first][1] if first < len(b["top2_logprobs"]) else float("nan")
        gap_other = d["top2_logprobs"][first][0] - d["top2_logprobs"][first][1] if first < len(d["top2_logprobs"]) else float("nan")
        verdict = "near-tie" if gap_stock <= 4 * max(noise, 1e-6) else "REAL DEVIATION"
        print(
            f"DIFFER  at token {first}  noise {noise:.2e}  stock top-2 gap {gap_stock:.2e}  "
            f"{label} top-2 gap {gap_other:.2e}  -> {verdict}  {b['prompt'][:50]!r}"
        )
        print(f"    stock: {b['text'][:160]!r}")
        print(f"    {label:5s} {d['text'][:160]!r}")
    print(f"{len(baseline) - mismatches}/{len(baseline)} prompts token-exact")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
