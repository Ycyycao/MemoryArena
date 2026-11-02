"""The gateway sets sampling parameters per role, the one place every domain's requests pass through."""
import json

from harness import gateway

CHAT = "/chat/completions"


def sent(role, body, rest=CHAT):
    return json.loads(gateway.apply_params(role, rest, json.dumps(body).encode()))


def test_by_default_the_agent_gets_temperature_one_over_what_it_passed(monkeypatch):
    monkeypatch.delenv("MARENA_GATEWAY_PARAMS", raising=False)
    assert sent("agent", {"model": "m", "temperature": 0}) == {"model": "m", "temperature": 1}


def test_by_default_other_roles_are_forwarded_byte_for_byte(monkeypatch):
    monkeypatch.delenv("MARENA_GATEWAY_PARAMS", raising=False)
    raw = b'{"temperature":  0.3}'
    assert gateway.apply_params("memory", CHAT, raw) == raw
    assert gateway.apply_params("judge", CHAT, raw) == raw


def test_the_setting_replaces_the_default(monkeypatch):
    monkeypatch.setenv("MARENA_GATEWAY_PARAMS", json.dumps(
        {"memory": {"max_tokens": 16384, "chat_template_kwargs": {"enable_thinking": False}}}))
    assert sent("memory", {"temperature": 0.3}) == {
        "temperature": 0.3, "max_tokens": 16384, "chat_template_kwargs": {"enable_thinking": False}}
    assert sent("agent", {"temperature": 0}) == {"temperature": 0}


def test_only_chat_completions_are_touched(monkeypatch):
    monkeypatch.delenv("MARENA_GATEWAY_PARAMS", raising=False)
    raw = b'{"input": ["x"]}'
    assert gateway.apply_params("agent", "/embeddings", raw) == raw
