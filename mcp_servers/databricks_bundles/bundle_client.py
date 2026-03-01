"""Databricks CLI bundle wrapper.

Provides thin, testable wrappers around the three Databricks CLI bundle
sub-commands used by the MCP server:

  bundle validate  – syntax-check a bundle
  bundle deploy    – create/update bundle resources in a workspace
  bundle run       – trigger a job or pipeline defined in a bundle

All commands require the ``databricks`` CLI to be on PATH and the workspace
host + token to be available via the standard environment variables:

  DATABRICKS_HOST   https://<workspace>.azuredatabricks.net
  DATABRICKS_TOKEN  personal access token
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


def _prepare_env() -> dict:
    """Return an env dict suitable for Databricks CLI subprocesses.

    The Databricks CLI reads ``DATABRICKS_TOKEN``.  This project stores the
    PAT as ``DATABRICKS_PAT`` (matching the credential naming convention in
    ``credentials.py``).  Bridge the two so callers only need to set one.
    """
    env = os.environ.copy()
    if not env.get("DATABRICKS_TOKEN") and env.get("DATABRICKS_PAT"):
        env["DATABRICKS_TOKEN"] = env["DATABRICKS_PAT"]
    return env


def _assert_cli_available() -> None:
    """Raise a clear RuntimeError if the Databricks CLI is not on PATH.

    Install instructions: https://docs.databricks.com/dev-tools/cli/install.html

    Quick install (Homebrew / pipx / direct):
        curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
    """
    if shutil.which("databricks") is None:
        raise RuntimeError(
            "Databricks CLI not found on PATH.  "
            "Install it with:\n"
            "  curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh\n"
            "or visit https://docs.databricks.com/dev-tools/cli/install.html"
        )


@dataclass
class BundleCommandResult:
    """Structured output from a Databricks CLI bundle command."""

    returncode: int
    stdout: str
    stderr: str
    succeeded: bool = field(init=False)

    def __post_init__(self) -> None:
        self.succeeded = self.returncode == 0

    def to_dict(self) -> dict:
        return {
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "succeeded": self.succeeded,
        }


def _run(cmd: list[str], cwd: str | Path | None = None, timeout: int = 300) -> BundleCommandResult:
    """Execute a CLI command and capture output."""
    _assert_cli_available()
    env = _prepare_env()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
        timeout=timeout,
    )
    return BundleCommandResult(
        returncode=result.returncode,
        stdout=result.stdout.strip(),
        stderr=result.stderr.strip(),
    )


def validate_bundle(bundle_path: str) -> BundleCommandResult:
    """Run ``databricks bundle validate`` in *bundle_path*.

    Args:
        bundle_path: Path to the directory containing ``databricks.yml``.

    Returns:
        BundleCommandResult with JSON output from the CLI on success.
    """
    return _run(["databricks", "bundle", "validate", "--output", "json"], cwd=bundle_path)


def deploy_bundle(bundle_path: str, target: str = "dev") -> BundleCommandResult:
    """Run ``databricks bundle deploy`` for the given *target*.

    Args:
        bundle_path: Path to the directory containing ``databricks.yml``.
        target:      Bundle target name (e.g. ``dev`` or ``prod``).

    Returns:
        BundleCommandResult with CLI stdout/stderr.
    """
    return _run(
        ["databricks", "bundle", "deploy", "--target", target],
        cwd=bundle_path,
        timeout=600,
    )


def run_bundle_resource(
    bundle_path: str,
    resource_key: str,
    target: str = "dev",
    extra_args: list[str] | None = None,
    only_tasks: list[str] | None = None,
    notebook_params: dict[str, str] | None = None,
) -> BundleCommandResult:
    """Run ``databricks bundle run`` for a job or pipeline resource.

    Args:
        bundle_path:     Path to the directory containing ``databricks.yml``.
        resource_key:    Bundle resource key, e.g. ``nyc_taxi_workflow`` or
                         ``nyc_taxi_dlt``.
        target:          Bundle target name.
        extra_args:      Additional CLI arguments (e.g. ``["--full-refresh"]``).
        only_tasks:      Comma-separated task keys to run (``--only`` flag).
        notebook_params: Key-value pairs forwarded as ``--notebook-params``.

    Returns:
        BundleCommandResult with run-id information in stdout.
    """
    cmd = ["databricks", "bundle", "run", "--target", target, resource_key]
    if only_tasks:
        cmd += ["--only", ",".join(only_tasks)]
    if notebook_params:
        pairs = ",".join(f"{k}={v}" for k, v in notebook_params.items())
        cmd += ["--notebook-params", pairs]
    if extra_args:
        cmd.extend(extra_args)
    return _run(cmd, cwd=bundle_path, timeout=1800)


def get_job_run_status(run_id: str) -> dict:
    """Fetch the status of a Databricks job run via the REST API.

    Falls back to the CLI ``databricks jobs get-run`` command so that the
    MCP server does not need an additional SDK dependency.

    Args:
        run_id: Databricks job run ID (integer as string).

    Returns:
        Dict with ``run_id``, ``state``, ``result_state``, and ``life_cycle_state``.
    """
    result = _run(["databricks", "jobs", "get-run", "--run-id", run_id])
    if not result.succeeded:
        return {
            "run_id": run_id,
            "error": result.stderr or "CLI command failed",
            "succeeded": False,
        }
    try:
        data = json.loads(result.stdout)
        state = data.get("state", {})
        return {
            "run_id": run_id,
            "life_cycle_state": state.get("life_cycle_state", "UNKNOWN"),
            "result_state": state.get("result_state", ""),
            "state_message": state.get("state_message", ""),
            "run_page_url": data.get("run_page_url", ""),
            "succeeded": True,
        }
    except json.JSONDecodeError:
        return {
            "run_id": run_id,
            "raw_output": result.stdout,
            "succeeded": True,
        }


def list_bundle_resources(bundle_path: str, target: str = "dev") -> BundleCommandResult:
    """Validate and list all resources defined in a bundle.

    Uses ``databricks bundle validate --output json`` and returns the parsed
    resources section for the requested target.

    Args:
        bundle_path: Path to the directory containing ``databricks.yml``.
        target:      Bundle target name.

    Returns:
        BundleCommandResult containing JSON output.
    """
    return _run(
        ["databricks", "bundle", "validate", "--target", target, "--output", "json"],
        cwd=bundle_path,
    )
