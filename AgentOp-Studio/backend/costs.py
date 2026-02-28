"""Cost estimation utilities for Mistral model runs."""

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
