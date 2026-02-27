"""Main entrypoint for the Mistral NYC hackathon agent.

This module wires together:
- Mistral LLM for reasoning
- NVIDIA NeMo Agent Toolkit for multi-step workflows
- ElevenLabs for voice output
- Weights & Biases for logging and evaluation

Copilot: when adding new functionality, prefer small, composable functions
and keep external API calls in agents.tools.
"""

import json
import os

from dotenv import load_dotenv

from agents import tools as tool_module
from agents.tools import get_tools

load_dotenv()

# Maximum number of tool-call rounds before stopping
_MAX_TOOL_ROUNDS = 5


def _dispatch_tool(name: str, arguments: str) -> str:
    """Invoke a tool function by name and return its result as a string.

    Args:
        name: Tool function name (must exist in agents.tools).
        arguments: JSON-encoded argument dictionary.

    Returns:
        String representation of the tool's return value.
    """
    func = getattr(tool_module, name, None)
    if func is None:
        return f"Unknown tool: {name}"
    kwargs = json.loads(arguments) if arguments else {}
    result = func(**kwargs)
    return str(result)


def run_agent(prompt: str) -> str:
    """Run the hackathon agent with the given user prompt.

    Executes an agentic loop: the model may request tool calls, which are
    dispatched and fed back as tool messages until the model produces a
    final text response or the round limit is reached.

    Args:
        prompt: The user's input message or task description.

    Returns:
        The agent's final response as a string.
    """
    tools = get_tools()

    # Import here to avoid hard dependency if mistralai is not installed
    from mistralai import Mistral  # noqa: PLC0415

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
    max_tokens = int(os.getenv("MAX_TOKENS", "2048"))
    temperature = float(os.getenv("TEMPERATURE", "0.6"))

    messages: list[dict] = [{"role": "user", "content": prompt}]

    for _ in range(_MAX_TOOL_ROUNDS):
        response = client.chat.complete(
            model=model,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        choice = response.choices[0]
        assistant_message = choice.message

        # If the model returned a final text answer, we're done.
        if not assistant_message.tool_calls:
            return assistant_message.content or ""

        # Append the assistant's tool-call request to the conversation.
        messages.append({"role": "assistant", "content": assistant_message.content, "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in assistant_message.tool_calls
        ]})

        # Execute each requested tool and append the results.
        for tc in assistant_message.tool_calls:
            tool_result = _dispatch_tool(tc.function.name, tc.function.arguments)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            })

    # If we exhausted tool rounds, do a final completion without tools.
    final_response = client.chat.complete(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return final_response.choices[0].message.content or ""


if __name__ == "__main__":
    import sys

    user_prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello, agent!"
    result = run_agent(user_prompt)
    print(result)
