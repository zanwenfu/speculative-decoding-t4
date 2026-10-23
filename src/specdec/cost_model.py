"""Closed-form cost models for speculative decoding.

All quantities are wall-clock milliseconds unless noted. No PyTorch /
CUDA dependency: pure arithmetic, safe to unit-test and pin.

Reference: Leviathan et al. 2023 (linear SD), Chen et al. 2024 (Sequoia
tree SD). The per-call decomposition `C = Tw + Tf` is this project's
contribution and is the operational version of the paper's closed-form result.
"""
from __future__ import annotations

from typing import Sequence


def closed_form_c(
    params_draft: float,
    params_target: float,
    bandwidth_gbps: float,
    fixed_overhead_ms: float,
    bytes_per_param: int = 2,
) -> float:
    """Predict the per-call cost ratio c = Cd/Ct from first principles.

    C = Tw + Tf where Tw = params * bytes / bandwidth is weight-loading
    time and Tf is per-call fixed overhead (kernel launch, layernorm,
    RoPE, residuals). Two regimes:

      * Tf << Tw: c -> params_d / params_t  (compute-bound; Sequoia-style
        assumptions hold).
      * Tf ~= Tw: c -> 1                    (bandwidth-bound; speculative
        decoding breaks).

    For T4 + Llama-3.2 1B/3B with bandwidth=300 GB/s and Tf=13 ms this
    predicts c ~= 0.60; measured c is 0.596.
    """
    tw_d = (params_draft * bytes_per_param) / (bandwidth_gbps * 1e9) * 1e3
    tw_t = (params_target * bytes_per_param) / (bandwidth_gbps * 1e9) * 1e3
    return (tw_d + fixed_overhead_ms) / (tw_t + fixed_overhead_ms)


def linear_sd_breakeven_alpha(k: int, c: float) -> float:
    """Break-even acceptance for linear SD at k draft tokens, in one common form.

    speedup > 1 iff alpha > (k*c + 1) / (k + 1), which holds when an iteration
    yields alpha * (k + 1) tokens. The alpha measured in this repository is
    defined by tokens per iteration = 1 + alpha * k; under that definition the
    break-even is alpha > c for every k. See ``src/bench/bench_linear.py``.
    """
    return (k * c + 1.0) / (k + 1.0)


def linear_sd_predicted_speedup(alpha: float, k: int, c: float) -> float:
    """Leviathan et al. 2023 Eq. 3 (greedy, geometric-length)."""
    if alpha == 1.0:
        return (k + 1) / (k * c + 1.0)
    numerator = 1.0 - alpha ** (k + 1)
    denominator = (1.0 - alpha) * (k * c + 1.0)
    return numerator / denominator


def tree_predicted_speedup(
    expected_tokens: float,
    depth: int,
    c: float,
    verify_cost: float,
) -> float:
    """Sequoia's hardware-aware speedup for one tree shape.

    ``expected_tokens`` is the tokens one iteration yields (from
    :func:`sequoia_dp`), ``verify_cost`` is t(n), the cost of verifying the
    tree in units of one single-token target call, and drafting costs
    ``depth`` draft calls of ``c`` each. This is the formula the phase 2B
    notebooks use.
    """
    return expected_tokens / (verify_cost + depth * c)


def sequoia_dp(
    sibling_accept_probs: Sequence[float],
    max_nodes: int,
    max_depth: int,
    max_branch: int = 5,
) -> dict[tuple[int, int], float]:
    """Sequoia's tree dynamic program (Chen et al. 2024, Alg. 4).

    ``sibling_accept_probs[k - 1]`` is the probability that the k-th child of
    an accepted node is the one accepted: the ``p_rank_b`` field of
    ``swor_acceptance_vector.json``. These sum to at most 1. The conditional
    rates in that file's ``acceptance_vector`` field (P(rank k | rank > k-1))
    are the wrong input: they sum to more than 1 and inflate the result.

    Returns ``{(nodes, depth): expected tokens per iteration}`` for the best
    tree with at most ``nodes`` nodes and ``depth`` levels, root included, so
    every value is at most ``depth``. Shapes with no valid tree are omitted.
    """
    neg = float("-inf")
    p = [0.0, *sibling_accept_probs][: max_branch + 1]
    p += [0.0] * (max_branch + 1 - len(p))
    # best_with[n][d][b]: best value of a tree of n nodes, depth <= d, whose
    # root has exactly b children; best[n][d] maximises over b.
    best_with = [[[neg] * (max_branch + 1) for _ in range(max_depth + 1)]
                 for _ in range(max_nodes + 1)]
    best = [[neg] * (max_depth + 1) for _ in range(max_nodes + 1)]
    for d in range(1, max_depth + 1):
        best_with[1][d][0] = best[1][d] = 1.0
    for n in range(2, max_nodes + 1):
        for d in range(2, max_depth + 1):
            for b in range(1, max_branch + 1):
                vals = [best_with[n - m][d][b - 1] + p[b] * best[m][d - 1]
                        for m in range(1, n)
                        if best_with[n - m][d][b - 1] > neg and best[m][d - 1] > neg]
                if vals:
                    best_with[n][d][b] = max(vals)
            best[n][d] = max(best_with[n][d])
    return {(n, d): best[n][d] for n in range(1, max_nodes + 1)
            for d in range(1, max_depth + 1) if best[n][d] > neg}
