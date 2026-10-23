"""Unit tests for the cost-model primitives.

These tests encode the paper's headline quantitative claims so that
regressions in the library show up as test failures.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from src.specdec.cost_model import (
    closed_form_c,
    linear_sd_breakeven_alpha,
    linear_sd_predicted_speedup,
    sequoia_dp,
    tree_predicted_speedup,
)

RESULTS = Path(__file__).resolve().parents[1] / "src" / "results"
ACCEPTANCE = json.loads((RESULTS / "swor_acceptance_vector.json").read_text())


def test_closed_form_c_matches_t4_measurement():
    # T4 HBM ~300 GB/s, Llama-3.2 1B/3B, Tf ~= 13 ms.
    # README ('A closed-form heuristic') predicts c ~= 0.60; measured 0.596.
    c = closed_form_c(
        params_draft=1_000_000_000,
        params_target=3_000_000_000,
        bandwidth_gbps=300,
        fixed_overhead_ms=13.0,
    )
    assert 0.58 <= c <= 0.62, f"closed-form c={c:.3f} outside [0.58, 0.62]"


def test_closed_form_c_compute_bound_limit():
    # Tf << Tw -> c -> params ratio. Set Tf tiny.
    c = closed_form_c(1_000_000_000, 3_000_000_000,
                      bandwidth_gbps=300, fixed_overhead_ms=0.001)
    assert abs(c - 1 / 3) < 0.01


def test_linear_sd_breakeven_k3_on_t4():
    # README Table 1: alpha_measured=0.732 just clears breakeven 0.697.
    alpha_star = linear_sd_breakeven_alpha(k=3, c=0.596)
    assert abs(alpha_star - 0.697) < 0.005


def test_linear_sd_predicted_speedup_bounds():
    # At alpha=1 the predicted speedup equals (k+1)/(kc+1).
    for k in (1, 3, 5, 10):
        pred = linear_sd_predicted_speedup(alpha=1.0, k=k, c=0.596)
        expected = (k + 1) / (k * 0.596 + 1)
        assert math.isclose(pred, expected, rel_tol=1e-6)


def test_sequoia_dp_is_monotonic_in_budget():
    table = sequoia_dp(ACCEPTANCE["p_rank_b"], max_nodes=8, max_depth=3)
    assert table[(8, 3)] >= table[(4, 3)] >= table[(1, 3)]
    assert table[(8, 3)] >= table[(6, 2)]


def test_sequoia_dp_never_yields_more_tokens_than_levels():
    table = sequoia_dp(ACCEPTANCE["p_rank_b"], max_nodes=32, max_depth=5)
    assert all(g <= d + 1e-9 for (_, d), g in table.items())


def test_sequoia_dp_predicts_the_measured_tokens_per_iteration():
    # The saved n=16, d=3 tree run accepted 2.95 tokens per iteration.
    tree = json.loads((RESULTS / "phase2b_tree_results.json").read_text())
    measured = next(s for s in tree["summaries"] if s["method"] == "tree_n16_d3")
    g = sequoia_dp(ACCEPTANCE["p_rank_b"], max_nodes=16, max_depth=3)[(16, 3)]
    assert abs(g - 2.941) < 0.001
    assert abs(g - measured["mean_tokens_per_iter"]) < 0.02


def test_conditional_rates_are_the_wrong_input():
    # P(rank k | rank > k-1) sums past 1 across siblings, so a depth-3 tree
    # would "yield" more than 3 tokens: the error behind the notebook's 1.68x.
    g = sequoia_dp(ACCEPTANCE["acceptance_vector"], max_nodes=16, max_depth=3)[(16, 3)]
    assert g > 3


def test_tree_predicted_speedup_arithmetic():
    assert math.isclose(tree_predicted_speedup(2.941, depth=3, c=0.596, verify_cost=1.15),
                        2.941 / (1.15 + 3 * 0.596))
