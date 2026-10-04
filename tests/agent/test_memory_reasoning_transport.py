"""Restricted requests cross the actual SDK/HTTP boundary once per reservation."""

import json
import threading
import time

import httpx
from openai import OpenAI, APIConnectionError
import pytest

from agent.memory_reasoning import CoreSingleAttemptTransport, PriceQuote, Route
from agent.memory_reasoning import SubscriptionAccountContract


class Agent:
    def __init__(self, mode, handler, provider="openai"):
        self.api_mode = mode
        self.provider = provider
        self.parent = object()
        self.parent_closed = False
        self.sends = 0
        self.aborts = 0
        self.releases = 0

        def counted(request):
            self.sends += 1
            return handler(request)

        self.http = httpx.Client(transport=httpx.MockTransport(counted))
        self.client = OpenAI(api_key="unit-test", base_url="https://example.test/v1",
                             http_client=self.http, max_retries=0)

    def _create_request_openai_client(self, **kwargs):
        assert self.client.max_retries == 0
        return self.client

    def _abort_request_openai_client(self, client, *, reason):
        assert client is self.client
        self.aborts += 1

    def _close_request_openai_client(self, client, *, reason):
        assert client is self.client
        self.releases += 1

    def _is_codex_backend(self):
        return False


def transport(*, cancelled=lambda: False, deadline=None, price=True, provider="openai"):
    return CoreSingleAttemptTransport(
        input_bound=lambda request, route: 100,
        approved_price_version="2026-10-test",
        price=PriceQuote("2026-10-test", provider, "gpt-5", 2, 8) if price else None,
        cancelled=cancelled, deadline_monotonic=deadline or time.monotonic() + 5,
    )


@pytest.mark.parametrize('invalid', ['missing', 'version', 'route', 'revoked', 'quota'])
def test_subscription_requires_explicit_current_host_approval(invalid):
    agent = Agent('codex_responses', lambda _: pytest.fail('unauthorised send'), provider='openai-codex')
    contract = SubscriptionAccountContract('test-account', 'openai-codex', 'gpt-5', 'codex_responses')
    single = CoreSingleAttemptTransport(input_bound=lambda *_: 100, price=None,
        cancelled=lambda: False, deadline_monotonic=time.monotonic() + 5,
        accounting_mode='subscription', subscription_contract=None if invalid == 'missing' else contract,
        approved_subscription_version='wrong' if invalid == 'version' else contract.version,
        subscription_approval_valid=lambda: invalid != 'revoked',
        quota_observation=lambda: {'state': 'REFUSED'} if invalid == 'quota' else {'state': 'UNKNOWN'})
    try:
        with pytest.raises(ValueError):
            single.complete({'model': 'wrong' if invalid == 'route' else 'gpt-5', 'input': []}, agent)
        assert agent.sends == 0
    finally:
        agent.client.close()


def test_chat_sdk_one_http_send_effort_usage_price_and_parent_preserved():
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path.endswith("/chat/completions")
        assert body["reasoning_effort"] == "high"
        return httpx.Response(200, json={"id": "chatcmpl-1", "object": "chat.completion",
            "created": 1, "model": "gpt-5", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}})

    agent = Agent("chat_completions", handler)
    route = Route("openai", "gpt-5", "high")
    wire = {"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "high"}
    single = transport()
    try:
        assert single.effective_effort(wire, route) == "high"
        response = single.complete(wire, agent)
        assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (11, 3)
        assert single.actual_cost(response, route) == pytest.approx((11 * 2 + 3 * 8) / 1_000_000)
        assert agent.sends == agent.releases == 1
        assert agent.aborts == 0 and not agent.parent_closed
    finally:
        agent.client.close()


def test_codex_responses_sdk_stream_core_parser_one_send():
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path.endswith("/responses")
        assert body["stream"] is True
        assert body["reasoning"]["effort"] == "high"
        events = [
            {"type": "response.output_item.done", "output_index": 0, "item": {
                "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": "ok"}]}},
            {"type": "response.completed", "response": {"id": "resp_1", "status": "completed",
                "usage": {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16}}},
        ]
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text="".join("data: " + json.dumps(e) + "\n\n" for e in events))

    agent = Agent("codex_responses", handler, provider="openai-codex")
    route = Route("openai-codex", "gpt-5", "high", api_mode="codex_responses")
    wire = {"model": "gpt-5", "instructions": "inspect", "input": [{"role": "user", "content": "hi"}],
            "reasoning": {"effort": "high"}, "store": False}
    single = transport(provider="openai-codex")
    try:
        assert single.effective_effort(wire, route) == "high"
        response = single.complete(wire, agent)
        assert response.output[0].content[0].text == "ok"
        assert (response.usage.input_tokens, response.usage.output_tokens) == (12, 4)
        assert single.actual_cost(response, route) == pytest.approx((12 * 2 + 4 * 8) / 1_000_000)
        assert agent.sends == agent.releases == 1
    finally:
        agent.client.close()


@pytest.mark.parametrize("mode", ["chat_completions", "codex_responses"])
def test_failed_http_send_has_no_hidden_retry(mode):
    agent = Agent(mode, lambda request: httpx.Response(503, json={"error": {"message": "unavailable"}}))
    wire = ({"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]}
            if mode == "chat_completions" else
            {"model": "gpt-5", "input": [{"role": "user", "content": "hi"}], "store": False})
    try:
        with pytest.raises(Exception):
            transport().complete(wire, agent)
        assert agent.sends == agent.releases == 1
    finally:
        agent.client.close()


def test_missing_versioned_price_fails_closed():
    route = Route("openai", "gpt-5", "high")
    assert transport(price=False).cost_upper_bound(100, 20, route) is None
    assert transport().cost_upper_bound(100, 20, Route("openai", "other", "high")) is None
    single = transport()
    single.effective_effort({"reasoning_effort": "high", "service_tier": "priority"}, route)
    assert single.cost_upper_bound(100, 20, route) is None


@pytest.mark.parametrize("stop", ["cancel", "deadline"])
def test_cancel_and_deadline_abort_only_request_client(stop):
    entered, release = threading.Event(), threading.Event()
    cancelled = threading.Event()

    def handler(request):
        entered.set()
        release.wait(2)
        raise httpx.ReadError("socket retired")

    agent = Agent("chat_completions", handler)
    original_abort = agent._abort_request_openai_client

    def abort(client, *, reason):
        original_abort(client, reason=reason)
        release.set()

    agent._abort_request_openai_client = abort
    deadline = time.monotonic() + (0.15 if stop == "deadline" else 5)
    single = transport(cancelled=cancelled.is_set, deadline=deadline)
    result = []
    worker = threading.Thread(target=lambda: result.append(_capture(lambda: single.complete(
        {"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]}, agent))))
    try:
        worker.start()
        assert entered.wait(1)
        if stop == "cancel":
            cancelled.set()
        worker.join(2)
        assert not worker.is_alive()
        assert isinstance(result[0], Exception)
        assert agent.aborts == agent.releases == agent.sends == 1
        assert not agent.parent_closed
    finally:
        release.set()
        agent.client.close()


def _capture(fn):
    try:
        return fn()
    except Exception as exc:
        return exc


@pytest.mark.parametrize("missing", ["price", "version", "stale_version", "bound", "zero", "bool", "fraction"])
def test_unapproved_cost_contract_never_reaches_http(missing):
    agent = Agent("chat_completions", lambda _: pytest.fail("unapproved dispatch"))
    single = transport()
    if missing == "price":
        single.price = None
    elif missing in {"version", "stale_version"}:
        single.approved_price_version = None if missing == "version" else "old-synthetic-version"
    else:
        single.input_bound = None if missing == "bound" else lambda *_: {
            "zero": 0, "bool": True, "fraction": 1.5}[missing]
    try:
        with pytest.raises(ValueError):
            single.complete({"model": "gpt-5", "messages": []}, agent)
        assert agent.sends == agent.releases == 0
    finally:
        agent.client.close()


@pytest.mark.parametrize("mode", ["chat_completions", "codex_responses"])
def test_whole_final_body_is_bounded_before_the_sdk_send(mode):
    captured = []
    def http_handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(503, json={"error": {"message": "synthetic failure"}})
    agent = Agent(mode, http_handler)
    single = transport()
    observed = []
    def bound(body, route):
        observed.append(json.loads(json.dumps(body)))
        body["tools"].clear()  # The estimator cannot alter the dispatched schema.
        return 100
    single.input_bound = bound
    wire = {"model": "gpt-5", "tools": [], "extra_body": {
        "tools": [{"type": "function", "function": {"name": "synthetic"}}],
        "synthetic_extension": {"evidence": "all input fields are bounded"}}}
    if mode == "chat_completions":
        wire.update(messages=[{"role": "system", "content": "host instructions"},
                              {"role": "tool", "tool_call_id": "prior", "content": "prior evidence"}],
                    max_completion_tokens=40)
    else:
        wire.update(instructions="host instructions", input=[{"role": "user", "content": "prior evidence"}],
                    max_output_tokens=40)
    try:
        with pytest.raises(Exception):
            single.complete(wire, agent)
        assert agent.sends == 1 and len(observed) == 1
        assert observed[0]["tools"] == captured[0]["tools"]
        assert observed[0]["synthetic_extension"] == captured[0]["synthetic_extension"]
        for field in ("messages",) if mode == "chat_completions" else ("instructions", "input"):
            assert observed[0][field] == captured[0][field]
        assert single.physical_output_cap(observed[0], Route("openai", "gpt-5", "high", mode, 40))
    finally:
        agent.client.close()


@pytest.mark.parametrize("cancel", [True, False])
@pytest.mark.parametrize("accounting_mode", ["metered", "subscription"])
def test_consumer_codex_has_no_physical_cap_and_late_terminal_usage_survives(cancel, accounting_mode):
    from agent.memory_reasoning.transport import CompletedUsageInterrupted
    stopped = threading.Event()
    class Stream:
        def __iter__(self):
            if cancel:
                stopped.set()
            yield SimpleNamespace(type="response.completed", response=SimpleNamespace(
                usage=SimpleNamespace(input_tokens=12, output_tokens=4000), service_tier="priority"))
        def close(self):
            pass
    from types import SimpleNamespace
    observed = []
    def create(**wire):
        observed.append(wire)
        return Stream()
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    agent = SimpleNamespace(provider="openai-codex", model="gpt-5", api_mode="codex_responses",
        _is_codex_backend=lambda: True, _create_request_openai_client=lambda **kw: client,
        _close_request_openai_client=lambda *a, **kw: None,
        _abort_request_openai_client=lambda *a, **kw: None)
    single = transport(provider="openai-codex", cancelled=stopped.is_set)
    if accounting_mode == 'subscription':
        single.accounting_mode = 'subscription'
        single.subscription_contract = SubscriptionAccountContract('synthetic-account', 'openai-codex', 'gpt-5', 'codex_responses')
        single.approved_subscription_version = 'subscription-token-v1'
        single.subscription_approval_valid = lambda: True
        single.price = None
    else:
        single.price = PriceQuote("2026-10-test", "openai-codex", "gpt-5", 2, 8, "priority")
    wire = {"model": "gpt-5", "input": [], "max_output_tokens": 40, "service_tier": "priority"}
    prepared = single.prepare_request(wire, agent)
    assert "max_output_tokens" not in prepared
    assert not single.physical_output_cap(prepared, Route("openai-codex", "gpt-5", "high", "codex_responses", 40))
    if cancel:
        with pytest.raises(CompletedUsageInterrupted) as caught:
            single.complete(prepared, agent)
        usage = caught.value.usage
    else:
        answer = single.complete(prepared, agent)
        usage = single._measured_usage(answer, Route("openai-codex", "gpt-5", "high", "codex_responses"))
    assert len(observed) == 1
    assert usage == {"input_tokens": 12, "output_tokens": 4000,
                                  "cost_usd": None if accounting_mode == 'subscription' else pytest.approx((12 * 2 + 4000 * 8) / 1_000_000)}


@pytest.mark.parametrize("failure", ["unapproved", "overbyte", "mutation", "sdk_mutation", "metered"])
def test_client_budget_rejects_before_http(failure):
    contract = SubscriptionAccountContract("approval", "openai", "gpt-5", "chat_completions",
        input_policy_version="client-budget-v1", max_wire_bytes=512, token_estimate=100,
        request_limit=1, deadline_seconds=5)
    single = CoreSingleAttemptTransport(input_bound=None, price=None,
        cancelled=lambda: False, deadline_monotonic=time.monotonic()+5,
        accounting_mode="metered" if failure == "metered" else "subscription",
        subscription_contract=contract, approved_subscription_version=contract.version,
        subscription_approval_valid=lambda: True,
        approved_input_policy_version=None if failure == "unapproved" else "client-budget-v1")
    route = Route("openai", "gpt-5", "medium")
    agent = Agent("chat_completions", lambda _: pytest.fail("forbidden send"))
    body = {"model": "gpt-5", "messages": [{"role": "user", "content": "synthetic"}],
            "reasoning_effort": "medium", "extra_body": {"instructions": "fixed"}}
    try:
        if failure == "overbyte":
            body["extra_body"]["instructions"] = "x"*513
        if failure in {"mutation", "sdk_mutation"}:
            prepared = single.prepare_request(body, agent)
            single.client_input_budget(prepared, route)
            if failure == "mutation":
                body["extra_body"]["instructions"] = "changed"
            else:
                def mutate(request):
                    request._content = b'{"instructions":"changed"}'
                agent.http.event_hooks["request"].append(mutate)
        with pytest.raises((ValueError, APIConnectionError)):
            single.complete(body, agent)
        assert agent.sends == 0
        if failure != "metered":
            with pytest.raises(ValueError):
                single.input_token_upper_bound({}, route)
    finally:
        agent.client.close()
