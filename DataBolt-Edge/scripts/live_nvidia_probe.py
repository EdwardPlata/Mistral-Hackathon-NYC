from __future__ import annotations

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from nvidia_api_management import run_probe  # noqa: E402


def main() -> int:
    result = run_probe(content="Return one short sentence about DataBolt Edge.")

    print(f"success={result.success}")
    print(f"latency_ms={result.latency_ms:.2f}")
    if result.status_code is not None:
        print(f"status_code={result.status_code}")
    if result.error:
        print(f"error={result.error}")
    if result.response:
        choices = result.response.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if content:
                print(f"response_preview={str(content)[:240]}")

    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
