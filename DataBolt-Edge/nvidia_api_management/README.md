# nvidia_api_management

Lightweight NVIDIA API testing package for `DataBolt-Edge` with:

- structured logging (with header redaction)
- retry + backoff (`requests` + `urllib3.Retry`)
- robust error handling
- environment-based credential and header management

## Package Layout

- `config.py`: env-driven runtime config
- `auth.py`: API key resolution + header construction
- `client.py`: resilient NVIDIA API client
- `errors.py`: package exceptions
- `logging_utils.py`: logger setup + header redaction
- `testing.py`: high-level probe helper (`run_probe`)

## Environment Variables

Required:

- `NVIDIA_API_KEY`: NVIDIA bearer token

Optional:

- `NVIDIA_API_BASE_URL` (default: `https://integrate.api.nvidia.com`)
- `NVIDIA_API_CHAT_PATH` (default: `/v1/chat/completions`)
- `NVIDIA_API_MODEL` (default: `mistralai/mistral-large-3-675b-instruct-2512`)
- `NVIDIA_API_TIMEOUT_SECONDS` (default: `30.0`)
- `NVIDIA_API_MAX_RETRIES` (default: `3`)
- `NVIDIA_API_BACKOFF_FACTOR` (default: `0.5`)
- `NVIDIA_API_RETRY_STATUS_CODES` (default: `429,500,502,503,504`)
- `NVIDIA_API_ACCEPT` (default: `application/json`)
- `NVIDIA_API_USER_AGENT` (default: `nvidia-api-management/0.1.0`)
- `NVIDIA_API_EXTRA_HEADERS` (JSON object string)

Example:

```bash
export NVIDIA_API_KEY="<your-token>"
export NVIDIA_API_MAX_RETRIES="4"
export NVIDIA_API_TIMEOUT_SECONDS="45"
```

## Quick Usage

From `DataBolt-Edge` app code:

```python
from nvidia_api_management import run_probe

result = run_probe(content="What is the capital of France?")
print(result.success, result.latency_ms, result.error)
```

From app script:

```bash
uv run python DataBolt-Edge/testing.py
```

## Run Tests

Unit tests (mocked HTTP):

```bash
uv run python -m unittest discover -s DataBolt-Edge/tests -p "test_nvidia_*.py"
```

Or run the included helper script:

```bash
bash DataBolt-Edge/scripts/run_nvidia_api_management_tests.sh
```

## Live Probe (Optional)

If `NVIDIA_API_KEY` is set:

```bash
uv run python DataBolt-Edge/scripts/live_nvidia_probe.py
```

This performs a real call and prints a summarized probe result.
