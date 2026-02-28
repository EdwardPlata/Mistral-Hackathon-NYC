"""Tests for cost estimation utilities.

conftest.py ensures AgentOp-Studio/ is on sys.path so ``backend`` is importable.
"""

from backend.costs import MISTRAL_PRICING, estimate_cost


def test_known_model_cost():
    # 1000 tokens on mistral-large-latest
    # average rate = (0.003 + 0.009) / 2 = 0.006 per 1000 tokens
    cost = estimate_cost(1000, "mistral-large-latest")
    assert abs(cost - 0.006) < 1e-9


def test_small_model_cheaper_than_large():
    cost_large = estimate_cost(1000, "mistral-large-latest")
    cost_small = estimate_cost(1000, "mistral-small-latest")
    assert cost_small < cost_large


def test_zero_tokens():
    assert estimate_cost(0, "mistral-large-latest") == 0.0


def test_unknown_model_falls_back_to_default():
    cost = estimate_cost(1000, "some-unknown-model-xyz")
    assert cost > 0


def test_all_known_models_positive():
    for model in MISTRAL_PRICING:
        assert estimate_cost(100, model) > 0


def test_proportional_scaling():
    cost_1k = estimate_cost(1000, "open-mistral-7b")
    cost_2k = estimate_cost(2000, "open-mistral-7b")
    assert abs(cost_2k - 2 * cost_1k) < 1e-12
