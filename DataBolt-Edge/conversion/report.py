"""Pipeline report generation for DataBolt Edge quantization pipeline.

Produces:
  - pipeline_report.json   — machine-readable full summary
  - pipeline_report.md     — human-readable Markdown comparison table

Usage::

    from conversion.report import generate_report, PipelineReport

    report = pipeline.run()          # QuantizationPipeline.run() returns PipelineReport
    generate_report(report, "/path/to/output/dir")

    print(report.summary_table())    # Prints a text table
    df = report.to_dataframe()       # Pandas DataFrame for further analysis
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .benchmark import BenchmarkResult
    from .pipeline import PipelineConfig, PipelineStep, StepOutcome
    from .quantize_onnx import QuantizationResult


# ── report dataclass ──────────────────────────────────────────────────────────

@dataclass
class PipelineReport:
    """Aggregated results from a :class:`~conversion.pipeline.QuantizationPipeline` run."""

    config: "PipelineConfig | None" = None
    outcomes: list["StepOutcome"] = field(default_factory=list)
    quant_results: list["QuantizationResult"] = field(default_factory=list)
    bench_results: list["BenchmarkResult"] = field(default_factory=list)

    # ── derived properties ────────────────────────────────────────────────────

    @property
    def succeeded(self) -> bool:
        """True if every executed step succeeded."""
        return all(o.success for o in self.outcomes)

    @property
    def successful_variants(self) -> list[str]:
        return [r.variant for r in self.quant_results if r.success]

    @property
    def failed_variants(self) -> list[str]:
        return [r.variant for r in self.quant_results if not r.success]

    # ── display helpers ───────────────────────────────────────────────────────

    def summary_table(self) -> str:
        """Return a compact ASCII comparison table."""
        lines: list[str] = []
        lines.append("=" * 72)
        lines.append("  DataBolt Edge — Quantization Pipeline Report")
        lines.append("=" * 72)

        if not self.quant_results and not self.bench_results:
            lines.append("  No results available.")
            return "\n".join(lines)

        # Quantization summary
        if self.quant_results:
            lines.append("")
            lines.append("  Quantization Results")
            lines.append("  " + "-" * 60)
            hdr = f"  {'Variant':<18} {'Source MB':>10} {'Output MB':>10} {'Ratio':>7} {'Time(s)':>8}  Status"
            lines.append(hdr)
            lines.append("  " + "-" * 60)
            for r in self.quant_results:
                if r.success:
                    row = (
                        f"  {r.variant:<18} "
                        f"{r.source_size_mb:>10.1f} "
                        f"{r.output_size_mb:>10.1f} "
                        f"{r.compression_ratio:>7.2f}x "
                        f"{r.duration_s:>8.1f}  OK"
                    )
                else:
                    row = f"  {r.variant:<18} {'':>10} {'':>10} {'':>7} {r.duration_s:>8.1f}  FAILED: {r.error[:30]}"
                lines.append(row)

        # Benchmark summary
        if self.bench_results:
            lines.append("")
            lines.append("  Benchmark Results")
            lines.append("  " + "-" * 60)
            hdr = f"  {'Variant':<18} {'Mean ms':>9} {'P50 ms':>8} {'P95 ms':>8} {'TPS':>8} {'MB':>7}"
            lines.append(hdr)
            lines.append("  " + "-" * 60)
            for r in self.bench_results:
                if r.success:
                    row = (
                        f"  {r.variant:<18} "
                        f"{r.mean_ms:>9.1f} "
                        f"{r.median_ms:>8.1f} "
                        f"{r.p95_ms:>8.1f} "
                        f"{r.tokens_per_second:>8.1f} "
                        f"{r.model_size_mb:>7.1f}"
                    )
                else:
                    row = f"  {r.variant:<18}  FAILED: {r.error[:40]}"
                lines.append(row)

            # Speedup vs fp32 baseline
            fp32 = next((r for r in self.bench_results if r.variant == "fp32" and r.success), None)
            if fp32 and fp32.mean_ms:
                lines.append("")
                lines.append("  Speedup vs fp32 baseline")
                lines.append("  " + "-" * 30)
                for r in self.bench_results:
                    if r.success and r.variant != "fp32":
                        speedup = fp32.mean_ms / r.mean_ms
                        lines.append(f"    {r.variant:<18} {speedup:>5.2f}x")

        lines.append("")
        lines.append("=" * 72)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a JSON-compatible dict."""
        return {
            "config": {
                "source_onnx": str(self.config.source_onnx) if self.config else "",
                "output_dir": str(self.config.output_dir) if self.config else "",
                "run_fp16": self.config.run_fp16 if self.config else None,
                "run_dynamic_int8": self.config.run_dynamic_int8 if self.config else None,
                "run_static_int8": self.config.run_static_int8 if self.config else None,
                "run_benchmark": self.config.run_benchmark if self.config else None,
            },
            "steps": [
                {
                    "step": o.step.name,
                    "success": o.success,
                    "duration_s": round(o.duration_s, 3),
                    "message": o.message,
                }
                for o in self.outcomes
            ],
            "quantization": [
                {
                    "variant": r.variant,
                    "success": r.success,
                    "source_size_mb": round(r.source_size_mb, 2),
                    "output_size_mb": round(r.output_size_mb, 2),
                    "compression_ratio": round(r.compression_ratio, 3),
                    "duration_s": round(r.duration_s, 2),
                    "output_path": str(r.output_path),
                    "error": r.error,
                }
                for r in self.quant_results
            ],
            "benchmarks": [r.as_dict() for r in self.bench_results],
            "summary": {
                "succeeded": self.succeeded,
                "successful_variants": self.successful_variants,
                "failed_variants": self.failed_variants,
            },
        }

    def to_dataframe(self):
        """Return a pandas DataFrame with benchmark results for all variants."""
        import pandas as pd  # type: ignore[import]
        if not self.bench_results:
            return pd.DataFrame()
        rows = []
        for r in self.bench_results:
            rows.append({
                "Variant": r.variant,
                "Mean (ms)": round(r.mean_ms, 1),
                "P50 (ms)": round(r.median_ms, 1),
                "P95 (ms)": round(r.p95_ms, 1),
                "TPS": round(r.tokens_per_second, 1),
                "Size (MB)": round(r.model_size_mb, 1),
                "Provider": r.provider,
                "OK": r.success,
            })
        return pd.DataFrame(rows)

    def to_markdown(self) -> str:
        """Return a Markdown report suitable for README / PR comments."""
        lines: list[str] = []
        lines.append("# DataBolt Edge — Quantization Pipeline Report")
        lines.append("")

        # Steps
        lines.append("## Pipeline Steps")
        lines.append("")
        lines.append("| Step | Status | Duration | Notes |")
        lines.append("|------|--------|----------|-------|")
        for o in self.outcomes:
            icon = "✅" if o.success else "❌"
            lines.append(
                f"| {o.step.name} | {icon} | {o.duration_s:.1f}s | {o.message[:60]} |"
            )
        lines.append("")

        # Quantization
        if self.quant_results:
            lines.append("## Quantization Results")
            lines.append("")
            lines.append("| Variant | Source MB | Output MB | Ratio | Time(s) |")
            lines.append("|---------|-----------|-----------|-------|---------|")
            for r in self.quant_results:
                if r.success:
                    lines.append(
                        f"| {r.variant} | {r.source_size_mb:.1f} | {r.output_size_mb:.1f} "
                        f"| {r.compression_ratio:.2f}x | {r.duration_s:.1f} |"
                    )
                else:
                    lines.append(f"| {r.variant} | — | — | — | FAILED: {r.error[:40]} |")
            lines.append("")

        # Benchmarks
        if self.bench_results:
            lines.append("## Benchmark Results")
            lines.append("")
            lines.append("| Variant | Mean (ms) | P50 (ms) | P95 (ms) | TPS | Size (MB) |")
            lines.append("|---------|-----------|----------|----------|-----|-----------|")
            fp32 = next((r for r in self.bench_results if r.variant == "fp32" and r.success), None)
            for r in self.bench_results:
                if r.success:
                    speedup = f"  ×{fp32.mean_ms / r.mean_ms:.2f}" if fp32 and r.variant != "fp32" else ""
                    lines.append(
                        f"| {r.variant}{speedup} | {r.mean_ms:.1f} | {r.median_ms:.1f} "
                        f"| {r.p95_ms:.1f} | {r.tokens_per_second:.1f} | {r.model_size_mb:.1f} |"
                    )

        return "\n".join(lines)


# ── file output ───────────────────────────────────────────────────────────────

def generate_report(report: PipelineReport, output_dir: str | Path) -> dict[str, Path]:
    """Write ``pipeline_report.json`` and ``pipeline_report.md`` to *output_dir*.

    Returns a dict mapping ``"json"`` and ``"markdown"`` to the written paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # JSON
    json_path = output_dir / "pipeline_report.json"
    with open(json_path, "w") as fh:
        json.dump(report.to_dict(), fh, indent=2, default=str)
    log.info("Report saved: %s", json_path)
    paths["json"] = json_path

    # Markdown
    md_path = output_dir / "pipeline_report.md"
    with open(md_path, "w") as fh:
        fh.write(report.to_markdown())
    log.info("Report saved: %s", md_path)
    paths["markdown"] = md_path

    return paths
