"""Databricks Asset Bundles MCP server.

Exposes five tools over the MCP stdio transport so that any MCP-compatible
client (including the project's Mistral agent) can deploy and operate
Databricks Asset Bundles without any Databricks SDK dependency in the agent.

Tools
-----
bundle_validate         – Syntax-check a local bundle directory.
bundle_deploy           – Create / update bundle resources in a workspace.
bundle_run              – Trigger a job or pipeline resource in a bundle.
bundle_get_run_status   – Poll the status of a Databricks job run.
bundle_list_resources   – List all resources declared in a bundle.

Transport
---------
Runs over stdin/stdout (MCP stdio transport).

Usage
-----
The server is launched as a subprocess by the Mistral agent tool wrapper:

    python -m mcp_servers.databricks_bundles.server

Environment variables required by the underlying Databricks CLI:
    DATABRICKS_HOST   – workspace URL
    DATABRICKS_TOKEN  – personal access token
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
)

from mcp_servers.databricks_bundles import bundle_client

# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------

app = Server("databricks-bundles")

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_TOOLS: list[Tool] = [
    Tool(
        name="bundle_validate",
        description=(
            "Syntax-check a Databricks Asset Bundle directory. "
            "Runs `databricks bundle validate` and returns the JSON output."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "bundle_path": {
                    "type": "string",
                    "description": "Absolute path to the directory containing databricks.yml.",
                },
                "target": {
                    "type": "string",
                    "description": "Bundle target to validate (default: dev).",
                    "default": "dev",
                },
            },
            "required": ["bundle_path"],
        },
    ),
    Tool(
        name="bundle_deploy",
        description=(
            "Deploy a Databricks Asset Bundle – creates or updates jobs, "
            "DLT pipelines, and other resources in the workspace. "
            "Runs `databricks bundle deploy`."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "bundle_path": {
                    "type": "string",
                    "description": "Absolute path to the directory containing databricks.yml.",
                },
                "target": {
                    "type": "string",
                    "description": "Bundle target to deploy to (e.g. dev, prod). Default: dev.",
                    "default": "dev",
                },
            },
            "required": ["bundle_path"],
        },
    ),
    Tool(
        name="bundle_run",
        description=(
            "Trigger a job or DLT pipeline defined in a Databricks Asset Bundle. "
            "Returns the run ID that can be polled with bundle_get_run_status. "
            "Use only_tasks to run specific tasks within a workflow job."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "bundle_path": {
                    "type": "string",
                    "description": "Absolute path to the directory containing databricks.yml.",
                },
                "resource_key": {
                    "type": "string",
                    "description": (
                        "Bundle resource key to run (e.g. nyc_taxi_workflow or nyc_taxi_dlt). "
                        "Matches the key under resources.jobs or resources.pipelines in databricks.yml."
                    ),
                },
                "target": {
                    "type": "string",
                    "description": "Bundle target (default: dev).",
                    "default": "dev",
                },
                "full_refresh": {
                    "type": "boolean",
                    "description": "Pass --full-refresh to the DLT pipeline run. Default: false.",
                    "default": False,
                },
                "only_tasks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Run only these task keys within the workflow (e.g. ['download_raw_data']).",
                },
                "notebook_params": {
                    "type": "object",
                    "description": "Key-value pairs to pass as notebook parameters (--notebook-params).",
                },
            },
            "required": ["bundle_path", "resource_key"],
        },
    ),
    Tool(
        name="bundle_get_run_status",
        description=(
            "Fetch the current status of a Databricks job run by its run ID. "
            "Returns life_cycle_state, result_state, and the run page URL."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "Databricks job run ID (integer as string).",
                },
            },
            "required": ["run_id"],
        },
    ),
    Tool(
        name="bundle_list_resources",
        description=(
            "List all resources (jobs, pipelines, etc.) declared in a Databricks "
            "Asset Bundle for a given target."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "bundle_path": {
                    "type": "string",
                    "description": "Absolute path to the directory containing databricks.yml.",
                },
                "target": {
                    "type": "string",
                    "description": "Bundle target (default: dev).",
                    "default": "dev",
                },
            },
            "required": ["bundle_path"],
        },
    ),
]


# ---------------------------------------------------------------------------
# Handler – list tools
# ---------------------------------------------------------------------------


@app.list_tools()
async def list_tools() -> list[Tool]:
    return _TOOLS


# ---------------------------------------------------------------------------
# Handler – call tool
# ---------------------------------------------------------------------------


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[TextContent]:
    args = arguments or {}
    try:
        result_data = _dispatch(name, args)
    except Exception as exc:  # noqa: BLE001
        result_data = {"error": str(exc), "succeeded": False}

    return [TextContent(type="text", text=json.dumps(result_data, indent=2))]


def _dispatch(name: str, args: dict[str, Any]) -> dict:
    """Route tool calls to bundle_client functions."""
    if name == "bundle_validate":
        res = bundle_client.validate_bundle(
            bundle_path=args["bundle_path"],
        )
        return res.to_dict()

    if name == "bundle_deploy":
        res = bundle_client.deploy_bundle(
            bundle_path=args["bundle_path"],
            target=args.get("target", "dev"),
        )
        return res.to_dict()

    if name == "bundle_run":
        extra: list[str] = []
        if args.get("full_refresh"):
            extra.append("--full-refresh")
        res = bundle_client.run_bundle_resource(
            bundle_path=args["bundle_path"],
            resource_key=args["resource_key"],
            target=args.get("target", "dev"),
            extra_args=extra or None,
            only_tasks=args.get("only_tasks") or None,
            notebook_params=args.get("notebook_params") or None,
        )
        # Try to surface the run_id from the CLI output
        output = res.to_dict()
        if res.succeeded:
            for line in res.stdout.splitlines():
                if "run_id" in line.lower() or line.strip().isdigit():
                    output["parsed_run_id"] = line.strip()
                    break
        return output

    if name == "bundle_get_run_status":
        return bundle_client.get_job_run_status(run_id=args["run_id"])

    if name == "bundle_list_resources":
        res = bundle_client.list_bundle_resources(
            bundle_path=args["bundle_path"],
            target=args.get("target", "dev"),
        )
        if res.succeeded:
            try:
                parsed = json.loads(res.stdout)
                return {"resources": parsed.get("resources", parsed), "succeeded": True}
            except json.JSONDecodeError:
                return res.to_dict()
        return res.to_dict()

    return {"error": f"Unknown tool: {name}", "succeeded": False}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def _main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    asyncio.run(_main())
