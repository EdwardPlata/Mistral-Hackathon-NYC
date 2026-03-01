from pprint import pprint

from nvidia_api_management import run_probe


def main() -> int:
    prompt = input("prompt: ")
    result = run_probe(content=prompt, stream=False)

    if result.success:
        print(f"Probe succeeded in {result.latency_ms:.2f}ms")
        pprint(result.response)
        return 0

    print(f"Probe failed in {result.latency_ms:.2f}ms")
    if result.status_code is not None:
        print(f"HTTP status: {result.status_code}")
    print(f"Error: {result.error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
