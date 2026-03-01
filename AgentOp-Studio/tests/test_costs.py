"""Tests for cost estimation utilities.

conftest.py ensures AgentOp-Studio/ is on sys.path so ``backend`` is importable.
"""

from backend.costs import (
    ELEVENLABS_PRICING,
    MISTRAL_PRICING,
    estimate_cost,
    estimate_elevenlabs_cost,
)


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


# ---------------------------------------------------------------------------
# ElevenLabs TTS cost tests
# ---------------------------------------------------------------------------


def test_elevenlabs_zero_chars():
    assert estimate_elevenlabs_cost(0) == 0.0


def test_elevenlabs_starter_tier_is_free():
    # Starter tier has 0.0 rate (free tier)
    assert estimate_elevenlabs_cost(5000, "starter") == 0.0


def test_elevenlabs_creator_tier():
    # 1000 chars at $0.24 per 1000 chars = $0.24
    cost = estimate_elevenlabs_cost(1000, "creator")
    assert abs(cost - 0.24) < 1e-9


def test_elevenlabs_higher_tier_cheaper_per_char():
    cost_creator = estimate_elevenlabs_cost(10_000, "creator")
    cost_pro = estimate_elevenlabs_cost(10_000, "pro")
    cost_scale = estimate_elevenlabs_cost(10_000, "scale")
    assert cost_pro < cost_creator
    assert cost_scale < cost_pro


def test_elevenlabs_unknown_tier_falls_back():
    cost = estimate_elevenlabs_cost(1000, "unknown-tier")
    assert cost > 0


def test_elevenlabs_all_known_tiers_non_negative():
    for tier in ELEVENLABS_PRICING:
        assert estimate_elevenlabs_cost(100, tier) >= 0


def test_elevenlabs_proportional_scaling():
    cost_1k = estimate_elevenlabs_cost(1000, "pro")
    cost_2k = estimate_elevenlabs_cost(2000, "pro")
    assert abs(cost_2k - 2 * cost_1k) < 1e-12
