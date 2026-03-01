"""Cost estimation utilities for Mistral model runs and ElevenLabs TTS."""

# USD per 1 000 tokens (input / output)
MISTRAL_PRICING: dict[str, dict[str, float]] = {
    "mistral-large-latest": {"input": 0.003, "output": 0.009},
    "mistral-medium-latest": {"input": 0.0027, "output": 0.0081},
    "mistral-small-latest": {"input": 0.001, "output": 0.003},
    "open-mistral-7b": {"input": 0.00025, "output": 0.00025},
    "open-mixtral-8x7b": {"input": 0.0007, "output": 0.0007},
    "open-mixtral-8x22b": {"input": 0.002, "output": 0.006},
    "codestral-latest": {"input": 0.001, "output": 0.003},
}

_DEFAULT_RATE = {"input": 0.003, "output": 0.009}

# ElevenLabs pricing: USD per 1 000 characters by subscription tier
# Based on published ElevenLabs pricing (2024)
ELEVENLABS_PRICING: dict[str, float] = {
    "starter": 0.0,     # Free tier — 10k chars/month included
    "creator": 0.24,    # $22/mo — ~92k chars effective rate
    "pro": 0.18,        # $99/mo — ~550k chars effective rate
    "scale": 0.12,      # $330/mo — ~2.75M chars effective rate
    "business": 0.08,   # $1320/mo — ~16.5M chars effective rate
}

_DEFAULT_ELEVENLABS_RATE = 0.30  # Conservative PAYG estimate per 1k chars


def estimate_cost(tokens: int, model: str = "mistral-large-latest") -> float:
    """Return a USD cost estimate for *tokens* total tokens on *model*.

    Uses the output rate as a conservative single-rate estimate when
    input/output split is unknown.

    Args:
        tokens: Total token count.
        model:  Mistral model name.

    Returns:
        Estimated cost in USD.
    """
    rates = MISTRAL_PRICING.get(model, _DEFAULT_RATE)
    # Use average of input+output as a rough single-rate estimate
    rate = (rates["input"] + rates["output"]) / 2
    return (tokens / 1000) * rate


def estimate_elevenlabs_cost(chars: int, tier: str = "creator") -> float:
    """Return a USD cost estimate for ElevenLabs TTS synthesis.

    Args:
        chars: Number of characters to synthesize.
        tier:  ElevenLabs subscription tier ('starter', 'creator', 'pro',
               'scale', 'business').

    Returns:
        Estimated cost in USD.
    """
    rate = ELEVENLABS_PRICING.get(tier, _DEFAULT_ELEVENLABS_RATE)
    return (chars / 1000) * rate
