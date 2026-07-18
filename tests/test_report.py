import json
from pathlib import Path
from types import SimpleNamespace

import httpx
from openai import OpenAI

from cascaderank.report import build_stable_graph_prefix
from cascaderank.report import generate_explanation


GRAPH_SUMMARY = {
    "dataset": "Tiny",
    "nodes": 6,
    "edges": 5,
    "connected_components": 1,
}
PAYLOAD = {
    "top_nodes": [
        {
            "node_id": "bridge",
            "gnn_rank": 1,
            "degree": 2,
            "betweenness": 0.8,
            "clustering": 0.0,
        }
    ],
    "structural_hole_candidate": {
        "node_id": "bridge",
        "degree": 2,
        "betweenness": 0.8,
        "clustering": 0.0,
        "neighbor_groups_without_node": 2,
        "is_articulation_point": True,
    },
    "early_attack_diagnostics": {
        "gap_at_5pct": 0.1,
        "gap_at_10pct": 0.2,
        "gnn_faster_than_random": True,
    },
}


class FakeResponses:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        details = SimpleNamespace(cached_tokens=1200, cache_write_tokens=0)
        usage = SimpleNamespace(input_tokens_details=details)
        return SimpleNamespace(
            id="resp_test",
            output_text="结论摘要\n\n节点 bridge 连接两个局部群。",
            usage=usage,
        )


def test_stable_prefix_is_long_and_deterministic() -> None:
    first = build_stable_graph_prefix(GRAPH_SUMMARY)
    second = build_stable_graph_prefix(dict(reversed(GRAPH_SUMMARY.items())))
    assert first == second
    assert len(first.encode("utf-8")) >= 7000


def test_openai_request_has_explicit_cache_boundary(tmp_path: Path) -> None:
    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    output = tmp_path / "explanation.txt"
    result = generate_explanation(
        GRAPH_SUMMARY,
        PAYLOAD,
        output,
        mode="openai",
        client=client,
    )

    assert result.provider == "openai"
    assert result.cached_tokens == 1200
    assert "结构洞" in output.read_text(encoding="utf-8")
    request = responses.kwargs
    assert request["model"] == "gpt-5.6-sol"
    assert request["prompt_cache_options"] == {
        "mode": "explicit",
        "ttl": "30m",
    }
    first_block = request["input"][0]["content"][0]
    assert first_block["prompt_cache_breakpoint"] == {"mode": "explicit"}


def test_openai_sdk_serializes_cache_request_and_parses_response(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "resp_sdk_test",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "error": None,
                "incomplete_details": None,
                "instructions": None,
                "max_output_tokens": 1800,
                "model": "gpt-5.6-sol",
                "output": [
                    {
                        "id": "msg_sdk_test",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "结构洞节点 bridge 连接两个分离邻域。",
                                "annotations": [],
                                "logprobs": [],
                            }
                        ],
                    }
                ],
                "parallel_tool_calls": True,
                "previous_response_id": None,
                "reasoning": {"effort": "low", "summary": None},
                "store": True,
                "temperature": 1.0,
                "text": {"format": {"type": "text"}},
                "tool_choice": "auto",
                "tools": [],
                "top_p": 1.0,
                "truncation": "disabled",
                "usage": {
                    "input_tokens": 1600,
                    "input_tokens_details": {
                        "cached_tokens": 1024,
                        "cache_write_tokens": 0,
                    },
                    "output_tokens": 100,
                    "output_tokens_details": {"reasoning_tokens": 10},
                    "total_tokens": 1700,
                },
                "metadata": {},
            },
        )

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    client = OpenAI(
        api_key="test-key",
        base_url="https://unit.test/v1",
        http_client=http_client,
    )
    try:
        result = generate_explanation(
            GRAPH_SUMMARY,
            PAYLOAD,
            tmp_path / "explanation.txt",
            mode="openai",
            client=client,
        )
    finally:
        client.close()

    assert result.response_id == "resp_sdk_test"
    assert result.model == "gpt-5.6-sol"
    assert result.cached_tokens == 1024
    assert captured["path"] == "/v1/responses"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "gpt-5.6-sol"
    assert body["reasoning"] == {"effort": "low"}
    assert body["prompt_cache_options"] == {
        "mode": "explicit",
        "ttl": "30m",
    }
    content = body["input"][0]["content"]
    assert content[0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert "Run-specific evidence follows" in content[1]["text"]


def test_auto_without_key_writes_labelled_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    output = tmp_path / "explanation.txt"
    result = generate_explanation(
        GRAPH_SUMMARY,
        PAYLOAD,
        output,
        mode="auto",
    )
    text = output.read_text(encoding="utf-8")
    assert result.provider == "local-fallback"
    assert "未调用 OpenAI API" in text
    assert "结构洞" in text
