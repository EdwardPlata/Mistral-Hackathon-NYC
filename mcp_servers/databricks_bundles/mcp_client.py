"""Stdio MCP client for the Databricks Bundles server.

Provides a single function – ``call_mcp_tool()`` – that launches the
``mcp_servers.databricks_bundles.server`` process, sends one JSON-RPC tool
call, and returns the parsed result.

The Mistral agent's ``agents/tools.py`` uses this module so that every
Databricks tool invocation goes through the full MCP protocol, making the
agent interchangeable with any other MCP-compatible host.

Usage (internal)::

    from mcp_servers.databricks_bundles.mcp_client import call_mcp_tool

    result = call_mcp_tool(
        "bundle_deploy",
        {"bundle_path": "/path/to/bundle", "target": "dev"},
    )
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


# Path to MCP server entry-point; keeps the client self-contained.
_SERVER_MODULE = "mcp_servers.databricks_bundles.server"


def _server_env() -> dict[str, str]:
    """Return the full current environment for the MCP server subprocess.

    Also bridges DATABRICKS_PAT → DATABRICKS_TOKEN so callers only need to
    set one credential name (matching the project convention in credentials.py).
    """
    env = {k: v for k, v in os.environ.items() if v is not None}
    if not env.get("DATABRICKS_TOKEN") and env.get("DATABRICKS_PAT"):
        env["DATABRICKS_TOKEN"] = env["DATABRICKS_PAT"]
    return env


async def _call_tool_async(tool_name: str, arguments: dict[str, Any]) -> dict:
    """Open an MCP session, call one tool, and return its result dict."""
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", _SERVER_MODULE],
        env=_server_env(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)

    # MCP returns a list of Content objects; extract the first text block.
    for content in result.content:
        if hasattr(content, "text"):
            try:
                return json.loads(content.text)
            except json.JSONDecodeError:
                return {"raw": content.text, "succeeded": True}
    return {"succeeded": False, "error": "No text content in MCP response"}


def call_mcp_tool(tool_name: str, arguments: dict[str, Any]) -> dict:
    """Synchronous wrapper around :func:`_call_tool_async`.

    Args:
        tool_name:  MCP tool name registered on the server.
        arguments:  Tool arguments dict.

    Returns:
        Parsed JSON dict from the server's text response.
    """
    return asyncio.run(_call_tool_async(tool_name, arguments))


def list_mcp_tools() -> list[dict]:
    """Return all tool schemas exposed by the MCP server."""

    async def _list() -> list[dict]:
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", _SERVER_MODULE],
            env=_server_env(),
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.inputSchema,
            }
            for t in result.tools
        ]

    return asyncio.run(_list())
