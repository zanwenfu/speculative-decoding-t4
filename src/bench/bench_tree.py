"""Tree speculative decoding on a T4: Sequoia's prediction against the saved measurements.

  * Default: prints a saved tree run (speedup, tokens per iteration and the
    per-iteration cost breakdown). No GPU needed.
  * --show-dp: also runs Sequoia's dynamic program on the measured
    per-sibling acceptance probabilities and prints its prediction.
  * --cuda: points to the notebook that runs the tree on a T4.

Examples
--------
    python -m src.bench.bench_tree --tree-config 16x3 --version v2 --show-dp
    python -m src.bench.bench_tree --tree-config 16x3 --version v3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.specdec.cost_model import sequoia_dp, tree_predicted_speedup

RESULTS = ROOT / "src" / "results"


def load_sibling_accept_probs() -> list[float]:
    """P(the k-th child is the accepted one), k = 1..5, from 305 samples."""
    return json.loads((RESULTS / "swor_acceptance_vector.json").read_text())["p_rank_b"]


def assumed_verify_cost(n: int) -> float:
    """The t(n) curve the phase 2B notebooks assumed, in single-call units."""
    if n <= 4:
        return 1.0
    if n <= 8:
        return 1.05
    if n <= 16:
        return 1.15
    if n <= 32:
        return 1.35
    return 1.70


def summarize_tree_result(version: str, n: int, d: int) -> None:
    path = RESULTS / (
        "phase2b_tree_results.json" if version == "v2"
        else "phase2b_v3_results.json"
    )
    data = json.loads(path.read_text())

    method = f"tree_n{n}_d{d}"
    summary = next(
        (s for s in data["summaries"] if s["method"] == method), None)
    if summary is None:
        print(f"No saved summary for {method} in {path.name}.", file=sys.stderr)
        print(f"Available: {[s['method'] for s in data['summaries']]}",
              file=sys.stderr)
        return

    profiler = data.get("profiler_snapshots", {}).get(f"n{n}_d{d}")
    baseline_tps = data["baseline_tps"]

    print(f"=== Tree SD {version}, (n={n}, d={d}) ===")
    print(f"Notebook prediction:   {summary['predicted_speedup']:.3f}x "
          "(fed conditional acceptance rates; see --show-dp)")
    print(f"Measured on T4:        {summary['speedup_vs_baseline']:.3f}x "
          f"({summary['mean_tok_sec']:.2f} tok/s vs baseline "
          f"{baseline_tps:.2f} tok/s)")
    print(f"Trials:                {summary['n_trials']} "
          f"(std {summary['std_tok_sec']:.2f} tok/s)")
    print(f"Tokens / iter (meas):  {summary['mean_tokens_per_iter']:.2f}")

    if profiler:
        print("\nPer-iteration decomposition (ms):")
        keys = [
            ("draft_prefix_ms_per_iter",  "draft prefix rebuild"),
            ("draft_extend_ms_per_iter",  "draft extends"),
            ("verify_ms_per_iter",        "target verify"),
            ("cache_extend_ms_per_iter",  "cache extend (v3 only)"),
        ]
        total = 0.0
        for k, label in keys:
            v = profiler.get(k)
            if v is None:
                continue
            print(f"  {label:<25s} {v:>8.2f}")
            total += v
        print(f"  {'iter total':<25s} {total:>8.2f}")


def show_dp(n: int, d: int, c: float, verify_cost: float | None) -> None:
    table = sequoia_dp(load_sibling_accept_probs(), max_nodes=n, max_depth=d)
    g = table[(n, d)]
    t = assumed_verify_cost(n) if verify_cost is None else verify_cost
    print(f"Sequoia DP, best tree with n<={n}, d<={d}: "
          f"{g:.3f} tokens per iteration (at most {d})")
    print(f"Predicted speedup:     {tree_predicted_speedup(g, d, c, t):.3f}x "
          f"(c={c:.3f}, t(n)={t:.2f})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tree-config", default="16x3",
                   help="<n>x<d>. Examples: 8x3, 16x3.")
    p.add_argument("--version", choices=("v2", "v3"), default="v2")
    p.add_argument("--from-results", action="store_true", default=True)
    p.add_argument("--cuda", action="store_true",
                   help="Delegate to notebook path (T4 required).")
    p.add_argument("--show-dp", action="store_true",
                   help="Print DP-optimal shape at the given budget.")
    p.add_argument("--c", type=float, default=0.596,
                   help="Measured Cd/Ct. Sequoia's A100 value is ~0.333.")
    p.add_argument("--verify-cost", type=float, default=None,
                   help="t(n) in single-call units. Default: the notebooks' assumed "
                        "curve (1.15 at n=16). Measured on the T4: 1.36 with a "
                        "cached prefix, 1.83 with the prompt re-encoded.")
    args = p.parse_args()

    n_str, d_str = args.tree_config.lower().split("x")
    n, d = int(n_str), int(d_str)

    if args.cuda:
        nb = ("phase2b_with_cd_probe.ipynb" if args.version == "v2"
              else "phase2b_v3_persistent_cache.ipynb")
        print(f"Full CUDA path: src/notebooks/{nb}. Open in Colab with a T4.",
              file=sys.stderr)
        return 2

    if args.show_dp:
        show_dp(n, d, args.c, args.verify_cost)
        print()
    summarize_tree_result(args.version, n, d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
