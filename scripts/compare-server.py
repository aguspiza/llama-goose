#!/usr/bin/env python3
"""Send the same requests to llama.cpp's llama-server and to the Goose
server, and compare the responses, less what changes from run to run (ids,
times, the build). Both must run the same model, with one slot:
    llama-server -m model.gguf --port 18431 -np 1
    tools/server.exe -m model.gguf --port 18432
    scripts/compare-server.py [ref_port] [goose_port]
Each request goes to both in turn, so the caches of the two see the same
prompts in the same order."""
import json
import sys
import urllib.request

REF = int(sys.argv[1]) if len(sys.argv) > 1 else 18431
GOOSE = int(sys.argv[2]) if len(sys.argv) > 2 else 18432

# cached_tokens: with SWA, llama-server reuses its cache only from a
# checkpoint; the Goose cache keeps every cell and reuses more
VOLATILE = {"created", "id", "timings", "system_fingerprint", "build_info", "cached_tokens"}


def call(port, method, path, body):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def clean(v):
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items() if k not in VOLATILE}
    if isinstance(v, list):
        return [clean(x) for x in v]
    return v


def parse(text, stream):
    if not stream:
        try:
            return clean(json.loads(text))
        except ValueError:
            return text
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block.startswith("data: "):
            continue
        payload = block[6:]
        events.append(payload if payload == "[DONE]" else clean(json.loads(payload)))
    return events


WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "The weather in a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}

CASES = [
    ("GET", "/health", None),
    ("GET", "/v1/models", None),
    ("POST", "/tokenize", {"content": "Hello world, this is a test"}),
    ("POST", "/tokenize", {"content": "Hello world", "with_pieces": True, "add_special": True}),
    ("POST", "/detokenize", {"tokens": [9259, 1902, 236764]}),
    ("POST", "/apply-template", {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}]}),
    ("POST", "/completion", {"prompt": "The sky is", "seed": 3, "n_predict": 8}),
    ("POST", "/completion", {"prompt": "The sky is", "seed": 3, "n_predict": 8}),
    ("POST", "/completion", {"prompt": "The sky is", "seed": 3, "n_predict": 30, "stop": ["and"]}),
    ("POST", "/completion", {"prompt": "Once upon a time", "temperature": 0, "n_predict": 24, "return_tokens": True}),
    ("POST", "/completion", {"prompt": "Once upon a time", "seed": 5, "n_predict": 12, "stream": True}),
    ("POST", "/completion", {"prompt": [2, 818, 7217, 563], "seed": 9, "n_predict": 6, "top_k": 5, "repeat_penalty": 1.1}),
    ("POST", "/completion", {"prompt": "List three fruits:", "temperature": 0, "n_predict": 40, "stop": ["\n\n"], "stream": True}),
    ("POST", "/completion", {"prompt": "JSON:", "temperature": 0, "n_predict": 30, "json_schema": {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}}),
    ("POST", "/v1/completions", {"prompt": "The sky is", "seed": 3, "max_tokens": 4}),
    ("POST", "/v1/completions", {"prompt": "The sky is", "seed": 3, "max_tokens": 4, "stream": True}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Say hi in three words."}], "seed": 7, "max_tokens": 20}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Say hi in three words."}], "seed": 7, "max_tokens": 6, "stream": True}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Count to five."}], "temperature": 0, "max_tokens": 30, "stream": True, "stream_options": {"include_usage": True}}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Give me a JSON object with a name."}], "temperature": 0, "max_tokens": 40, "response_format": {"type": "json_object"}}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi! How"}], "temperature": 0, "max_tokens": 10}),
    ("POST", "/v1/chat/completions", {"messages": [], "seed": 1}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "n_predict": -5}),
    ("POST", "/completion", {"prompt": "x", "n_predict": 2, "temperature": "hot"}),
    ("GET", "/nothing", None),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "What is the weather in Paris?"}], "tools": [WEATHER], "temperature": 0, "max_tokens": 60}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "What is the weather in Paris?"}], "tools": [WEATHER], "temperature": 0, "max_tokens": 60, "stream": True}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Weather in Rome?"}], "tools": [WEATHER], "tool_choice": "required", "temperature": 0, "max_tokens": 60}),
    ("POST", "/v1/chat/completions", {"messages": [
        {"role": "user", "content": "Weather in Rome?"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "id": "c1", "function": {"name": "get_weather", "arguments": {"city": "Rome"}}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "{\"temp\": 21}"}], "tools": [WEATHER], "temperature": 0, "max_tokens": 40}),
]


def main():
    failed = 0
    for method, path, body in CASES:
        stream = bool(body and body.get("stream"))
        rs, rt = call(REF, method, path, body)
        gs, gt = call(GOOSE, method, path, body)
        r, g = parse(rt, stream), parse(gt, stream)
        same = rs == gs and r == g
        label = f"{method} {path} {json.dumps(body)[:70] if body else ''}"
        print(("ok    " if same else "DIFF  ") + label)
        if not same:
            failed += 1
            print(f"  ref   ({rs}): {json.dumps(r)[:1500]}")
            print(f"  goose ({gs}): {json.dumps(g)[:1500]}")
    print(f"{len(CASES) - failed}/{len(CASES)} the same")
    sys.exit(1 if failed else 0)


main()
