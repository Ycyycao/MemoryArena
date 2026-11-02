"""LLM 调用的退避重试。

加这层的起因：`travel-simplemem-case0-8p-react2` 第 6 轮一个 400 掀掉了整条 run。
那条请求的内容是一段旅行推理加一张航班表，32k 字符，4.9 秒快速失败，错误串尾巴
带 `ip:port`——不是真的内容策略，是中转站在几条 run 并行时的误报。

全部零 API 开销：客户端的 `create` 换成假的。
"""

import pytest

from env.env_systems.travel_planner_env.clients.openai_client import OpenAIClient


class Boom(Exception):
    pass


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    return OpenAIClient(model_name="gpt-5-mini")


@pytest.fixture
def slept(monkeypatch):
    """退避不真睡，只记时长。"""
    delays = []
    monkeypatch.setattr("time.sleep", lambda d: delays.append(d))
    return delays


def scripted(client, monkeypatch, *outcomes):
    """按脚本回应：异常就抛，别的就返回。返回调用计数器。"""
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        item = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(client.client.chat.completions, "create", fake)
    return calls


def test_success_on_the_first_try_does_not_retry(client, monkeypatch, slept):
    calls = scripted(client, monkeypatch, "ok")
    assert client._call(model="m", messages=[]) == "ok"
    assert len(calls) == 1
    assert client.retry_count == 0
    assert slept == []


def test_retries_then_succeeds(client, monkeypatch, slept):
    """react2 那次的形状：前面失败，后面就好了。"""
    calls = scripted(client, monkeypatch, Boom("400 content policy"), Boom("400"), "ok")
    assert client._call(model="m", messages=[]) == "ok"
    assert len(calls) == 3
    assert client.retry_count == 2
    assert len(slept) == 2


def test_backoff_grows_and_is_jittered(client, monkeypatch, slept):
    scripted(client, monkeypatch, Boom("x"), Boom("x"), "ok")
    client._call(model="m", messages=[])
    base = client.RETRY_BASE_SECONDS
    # 指数 + 抖动：第 n 次落在 base*2^(n-1) 的 [0.5, 1.5) 倍区间
    assert base * 0.5 <= slept[0] < base * 1.5
    assert base * 2 * 0.5 <= slept[1] < base * 2 * 1.5
    assert slept[1] > slept[0]           # 真的在变长，不是每次都等一样久


def test_exhausting_retries_reraises_the_last_error(client, monkeypatch, slept):
    """重试不是兜底：耗尽了还是要抛，让上层的 round.error 接。"""
    calls = scripted(client, monkeypatch, Boom("still broken"))
    with pytest.raises(Boom, match="still broken"):
        client._call(model="m", messages=[])
    assert len(calls) == client.RETRY_ATTEMPTS
    assert client.retry_count == client.RETRY_ATTEMPTS
    assert len(slept) == client.RETRY_ATTEMPTS - 1     # 最后一次失败后不睡


def test_retry_count_is_visible_in_usage_stats(client, monkeypatch, slept):
    """重试次数必须露出来——它是"中转站在限速"的唯一可见证据。

    重试最大的代价是把限速信号悄悄吃掉，那样就没法判断该不该少跑几条并行。
    """
    scripted(client, monkeypatch, Boom("x"), "ok")
    client._call(model="m", messages=[])
    assert client.get_usage_stats()["llm_retries"] == 1


def test_attempts_are_configurable(monkeypatch, slept):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(OpenAIClient, "RETRY_ATTEMPTS", 1)
    c = OpenAIClient(model_name="gpt-5-mini")
    calls = scripted(c, monkeypatch, Boom("x"))
    with pytest.raises(Boom):
        c._call(model="m", messages=[])
    assert len(calls) == 1               # 关掉重试就是只打一次
    assert slept == []
