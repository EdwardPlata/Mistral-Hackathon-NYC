"""Instrumented version of agents/main_agent.py.

Re-implements the run_agent() agentic loop with full observability:
every Mistral call, tool dispatch, token count, and cost is persisted
to DuckDB via backend.db.
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv

from agents import tools as tool_module
from agents.tools import get_tools

from .costs import estimate_cost
from .db import get_conn, init_schema

load_dotenv()

_MAX_TOOL_ROUNDS = 5


def _log_to_wandb(run_id: str, model: str, total_tokens: int, total_cost: float, status: str) -> None:
    """Log run metrics to Weights & Biases if WANDB_API_KEY is configured."""
    if not os.getenv("WANDB_API_KEY"):
        return
    try:
        import wandb  # noqa: PLC0415

        project = os.getenv("WANDB_PROJECT", "agentops-studio")
        if wandb.run is None:
            wandb.init(project=project, id=run_id, name=run_id, resume="allow")
        wandb.log(
            {
                "total_tokens": total_tokens,
                "total_cost_usd": total_cost,
                "status": 1 if status == "success" else 0,
                "model": model,
            }
        )
        wandb.finish()
    except Exception:
        pass  # W&B logging is best-effort


def _dispatch_tool(name: str, arguments: str) -> str:
    """Invoke a tool by name and return the result as a string."""
    func = getattr(tool_module, name, None)
    if func is None:
        return f"Unknown tool: {name}"
    kwargs = json.loads(arguments) if arguments else {}
    return str(func(**kwargs))


def run_instrumented(
    prompt: str,
    user_id: str = "default",
    agent_id: str = "main",
) -> dict:
    """Run the agent with full observability logging to DuckDB.

    Args:
        prompt:   User input / task description.
        user_id:  Caller identifier (for multi-user setups).
        agent_id: Logical agent name.

    Returns:
        dict with keys: run_id, response, total_tokens, total_cost
    """
    init_schema()

    run_id = str(uuid.uuid4())
    model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
    max_tokens = int(os.getenv("MAX_TOKENS", "2048"))
    temperature = float(os.getenv("TEMPERATURE", "0.6"))
    config = json.dumps(
        {"model": model, "max_tokens": max_tokens, "temperature": temperature}
    )
    start_time = datetime.now(timezone.utc)

    conn = get_conn()
    conn.execute(
        """
        INSERT INTO runs
            (run_id, agent_id, user_id, start_time, status, config, initial_prompt)
        VALUES (?, ?, ?, ?, 'running', ?, ?)
        """,
        [run_id, agent_id, user_id, start_time, config, prompt],
    )
    conn.close()

    messages_log: list[dict] = []
    tool_calls_log: list[dict] = []

    total_tokens = 0
    final_response = ""
    status = "success"

    try:
        from mistralai import Mistral  # noqa: PLC0415

        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
                "MISTRAL_API_KEY is not set. Please set it via environment variable "
                "or .env file. See docs/CREDENTIALS.md for setup instructions."
            )

        client = Mistral(api_key=api_key)
        tools = get_tools()
        messages: list[dict] = [{"role": "user", "content": prompt}]

        # Log the initial user message
        messages_log.append(
            {
                "message_id": str(uuid.uuid4()),
                "run_id": run_id,
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now(timezone.utc),
                "token_count": None,
                "finish_reason": None,
            }
        )

        for _ in range(_MAX_TOOL_ROUNDS):
            response = client.chat.complete(
                model=model,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            usage = response.usage
            if usage:
                total_tokens += usage.total_tokens or 0

            choice = response.choices[0]
            assistant_message = choice.message

            if not assistant_message.tool_calls:
                final_response = assistant_message.content or ""
                messages_log.append(
                    {
                        "message_id": str(uuid.uuid4()),
                        "run_id": run_id,
                        "role": "assistant",
                        "content": final_response,
                        "timestamp": datetime.now(timezone.utc),
                        "token_count": usage.total_tokens if usage else None,
                        "finish_reason": choice.finish_reason,
                    }
                )
                break

            # Record assistant tool-call message
            tc_content = json.dumps(
                [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    for tc in assistant_message.tool_calls
                ]
            )
            assistant_msg_id = str(uuid.uuid4())
            messages_log.append(
                {
                    "message_id": assistant_msg_id,
                    "run_id": run_id,
                    "role": "assistant",
                    "content": tc_content,
                    "timestamp": datetime.now(timezone.utc),
                    "token_count": usage.total_tokens if usage else None,
                    "finish_reason": choice.finish_reason,
                }
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in assistant_message.tool_calls
                    ],
                }
            )

            for tc in assistant_message.tool_calls:
                t0 = time.perf_counter()
                tool_result = _dispatch_tool(tc.function.name, tc.function.arguments)
                latency_ms = int((time.perf_counter() - t0) * 1000)

                tool_calls_log.append(
                    {
                        "call_id": str(uuid.uuid4()),
                        "run_id": run_id,
                        "message_id": assistant_msg_id,
                        "tool_name": tc.function.name,
                        "args": tc.function.arguments,
                        "return_value": tool_result,
                        "latency_ms": latency_ms,
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    }
                )
                messages_log.append(
                    {
                        "message_id": str(uuid.uuid4()),
                        "run_id": run_id,
                        "role": "tool",
                        "content": tool_result,
                        "timestamp": datetime.now(timezone.utc),
                        "token_count": None,
                        "finish_reason": None,
                    }
                )
        else:
            # Exhausted tool rounds — do a final completion without tools
            final_resp = client.chat.complete(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            usage = final_resp.usage
            if usage:
                total_tokens += usage.total_tokens or 0
            final_response = final_resp.choices[0].message.content or ""
            messages_log.append(
                {
                    "message_id": str(uuid.uuid4()),
                    "run_id": run_id,
                    "role": "assistant",
                    "content": final_response,
                    "timestamp": datetime.now(timezone.utc),
                    "token_count": usage.total_tokens if usage else None,
                    "finish_reason": final_resp.choices[0].finish_reason,
                }
            )

    except Exception as exc:
        status = "error"
        final_response = f"Error: {exc}"

    end_time = datetime.now(timezone.utc)
    total_cost = estimate_cost(total_tokens, model)

    # Bulk-insert messages + tool calls, update run record
    conn = get_conn()
    for msg in messages_log:
        conn.execute(
            """
            INSERT INTO messages
                (message_id, run_id, role, content, timestamp, token_count, finish_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                msg["message_id"],
                msg["run_id"],
                msg["role"],
                msg["content"],
                msg["timestamp"],
                msg["token_count"],
                msg["finish_reason"],
            ],
        )
    for tc in tool_calls_log:
        conn.execute(
            """
            INSERT INTO tool_calls
                (call_id, run_id, message_id, tool_name, args, return_value, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                tc["call_id"],
                tc["run_id"],
                tc["message_id"],
                tc["tool_name"],
                tc["args"],
                tc["return_value"],
                tc["latency_ms"],
            ],
        )
    conn.execute(
        """
        UPDATE runs
        SET end_time       = ?,
            status         = ?,
            total_tokens   = ?,
            total_cost     = ?,
            final_response = ?
        WHERE run_id = ?
        """,
        [end_time, status, total_tokens, total_cost, final_response, run_id],
    )
    conn.close()

    _log_to_wandb(run_id, model, total_tokens, total_cost, status)

    return {
        "run_id": run_id,
        "response": final_response,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
    }
