import os

import requests

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
stream = True

api_key = os.environ.get("NVIDIA_BEARER_TOKEN") or os.environ.get("NVIDIA_API_KEY")
if not api_key:
    raise RuntimeError("Set NVIDIA_BEARER_TOKEN or NVIDIA_API_KEY before running this script.")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Accept": "text/event-stream" if stream else "application/json",
}

payload = {
    "model": "mistralai/mistral-large-3-675b-instruct-2512",
    "messages": [{"role": "user", "content": "Hello, what can you do?"}],
    "max_tokens": 2048,
    "temperature": 0.15,
    "top_p": 1.00,
    "frequency_penalty": 0.00,
    "presence_penalty": 0.00,
    "stream": stream,
}

response = requests.post(invoke_url, headers=headers, json=payload)

if stream:
    for line in response.iter_lines():
        if line:
            print(line.decode("utf-8"))
else:
    print(response.json())
