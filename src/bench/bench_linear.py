"""Linear speculative decoding on a T4: measured acceptance against a per-call cost model.

Prints, for each draft length k, the acceptance measured in
``spec_decoding_phase1c.ipynb`` (alpha, defined by tokens per iteration =
1 + alpha * k), the speedup a per-call cost model predicts from it,
(1 + alpha * k) / (k * c + 1), and the measured speedup from
``spec_decoding_phase1.ipynb``. Under this model linear SD breaks even when
alpha > c. No GPU needed.

    python -m src.bench.bench_linear --k 3 5 7 10

The GPU sweep itself lives in src/notebooks/spec_decoding_phase1.ipynb.
"""
from __future__ import annotations

import argparse

# Measured on the T4 (Llama-3.2-1B drafting for 3B, fp16, greedy).
T4_LLAMA32 = {
    "c": 0.596,
    "alpha_measured": {3: 0.732, 5: 0.644, 7: 0.550, 10: 0.488},
    "speedup_measured": {3: 1.01, 5: 0.98, 7: 0.97, 10: 0.92},
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--k", nargs="+", type=int, default=[3, 5, 7, 10])
    p.add_argument("--c", type=float, default=T4_LLAMA32["c"],
                   help="Per-call cost ratio Cd/Ct. Default is the T4 measurement.")
    args = p.parse_args()

    print(f"{'k':>3} {'alpha':>7} {'tokens/iter':>12} {'model':>7} {'measured':>9}")
    print("-" * 42)
    for k in args.k:
        alpha = T4_LLAMA32["alpha_measured"].get(k)
        meas = T4_LLAMA32["speedup_measured"].get(k)
        if alpha is None:
            print(f"{k:>3} {'-':>7} {'-':>12} {'-':>7} {'-':>9}")
            continue
        tokens = 1 + alpha * k
        model = tokens / (k * args.c + 1)
        print(f"{k:>3} {alpha:>7.3f} {tokens:>12.2f} {model:>6.2f}x {meas:>8.2f}x")
    print(f"\nBreak-even under this model: alpha > c = {args.c:.3f} at every k.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
