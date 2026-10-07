"""One physical OpenAI-wire request for an isolated reasoning job."""

from __future__ import annotations

from dataclasses import dataclass
import json
import inspect
import threading
import time
from typing import Any, Callable, Mapping
from types import SimpleNamespace

from agent.codex_runtime import _consume_codex_event_stream, _sanitize_consumer_codex_request
from agent.sdk_transform_bypass import bypass_sdk_request_transform
from agent.chat_completion_helpers import bypass_chat_sdk_request_transform


def _tier(value: Any) -> str | None:
    return None if value in (None, "auto", "default") else str(value)


@dataclass(frozen=True)
class PriceQuote:
    version: str
    provider: str
    model: str
    input_usd_per_million: float
    output_usd_per_million: float
    service_tier: str | None = None

    def __post_init__(self):
        import math
        if not all(isinstance(v, str) and v.strip() for v in (self.version, self.provider, self.model)) or (self.service_tier is not None and
                not (isinstance(self.service_tier, str) and self.service_tier.strip())) or any(
            not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0
            for v in (self.input_usd_per_million, self.output_usd_per_million)
        ):
            raise ValueError("versioned, finite route pricing required")


class CompletedUsageInterrupted(TimeoutError):
    """A completed response was interrupted; retain only measured numeric usage."""

    def __init__(self, usage: Mapping[str, int | float | None]):
        super().__init__("reasoning request cancelled or deadline exceeded")
        self.usage = dict(usage)


@dataclass(frozen=True)
class SubscriptionAccountContract:
    """Host-owned approval identity; never derived from a failed price lookup."""

    contract_id: str
    provider: str
    model: str
    api_mode: str
    version: str = "subscription-token-v1"
    input_policy_version: str = "strict-input-v1"
    max_wire_bytes: int | None = None
    token_estimate: int | None = None
    request_limit: int | None = None
    deadline_seconds: int | None = None

    def __post_init__(self):
        if (self.version != "subscription-token-v1" or
                not all(isinstance(v, str) and v.strip() for v in
                        (self.contract_id, self.provider, self.model)) or
                self.api_mode not in {"chat_completions", "codex_responses"}):
            raise ValueError("explicit versioned subscription account contract required")
        if self.input_policy_version not in {"strict-input-v1", "client-budget-v1"}:
            raise ValueError("unsupported subscription input policy")
        values = (self.max_wire_bytes, self.token_estimate, self.request_limit, self.deadline_seconds)
        if self.input_policy_version == "client-budget-v1":
            if any(type(v) is not int or v <= 0 for v in values):
                raise ValueError("finite client budget approval required")
        elif any(v is not None for v in values):
            raise ValueError("client limits require explicit client-budget-v1")


class CoreSingleAttemptTransport:
    """Use Core's resolved route and request slot, with no Relay or SDK retry.

    ``input_bound`` is supplied by the host and must include the whole wire request,
    including tool schemas. Metered sends require exact approved pricing;
    subscription sends require an independently approved account contract.
    """

    single_attempt = True

    def __init__(self, *, input_bound: Callable[[Mapping[str, Any], Any], int] | None,
                 price: PriceQuote | None, cancelled: Callable[[], bool], deadline_monotonic: float,
                 approved_price_version: str | None = None,
                 accounting_mode: str = "metered",
                 subscription_contract: SubscriptionAccountContract | None = None,
                 approved_subscription_version: str | None = None,
                 subscription_approval_valid: Callable[[], bool] | None = None,
                 approved_input_policy_version: str | None = None,
                 quota_observation: Callable[[], Mapping[str, Any]] | None = None):
        self.approved_input_policy_version = approved_input_policy_version
        self._dispatches = 0
        self._created_at = time.monotonic()
        self.input_bound = input_bound
        self.price = price
        self.approved_price_version = approved_price_version
        if accounting_mode not in {"metered", "subscription"}:
            raise ValueError("unsupported accounting mode")
        self.accounting_mode = accounting_mode
        self.subscription_contract = subscription_contract
        self.approved_subscription_version = approved_subscription_version
        self.subscription_approval_valid = subscription_approval_valid
        self.quota_observation = quota_observation
        self.cancelled = cancelled
        self.deadline = deadline_monotonic
        self._requested_tier: str | None = None
        self._bound_request: tuple[str, int] | None = None
        self._consumer_codex = False

    def accounting_contract(self, route) -> dict[str, Any]:
        from dataclasses import asdict
        if self.accounting_mode == "metered":
            return {"accounting_mode": "metered", "accounting_version": "metered-price-v1"}
        contract = self.subscription_contract
        if (not isinstance(contract, SubscriptionAccountContract) or
                contract.version != self.approved_subscription_version or
                (contract.provider, contract.model, contract.api_mode) !=
                (route.provider, route.model, route.api_mode) or
                not callable(self.subscription_approval_valid) or
                self.subscription_approval_valid() is not True):
            raise ValueError("host-approved subscription contract unavailable or revoked")
        if contract.input_policy_version == "client-budget-v1" and (
                self.approved_input_policy_version != "client-budget-v1"):
            raise ValueError("explicit host input policy approval required")
        quota = {"state": "UNKNOWN", "remaining": None}
        if self.quota_observation is not None:
            observed = self.quota_observation()
            if not isinstance(observed, Mapping) or observed.get("state") not in {"UNKNOWN", "KNOWN", "REFUSED"}:
                raise ValueError("invalid observable quota")
            if observed["state"] == "REFUSED":
                raise ValueError("subscription quota refused")
            if observed["state"] == "KNOWN":
                remaining = observed.get("remaining")
                import math
                if (type(remaining) not in (int, float) or not math.isfinite(remaining)
                        or remaining <= 0 or not observed.get("source") or not observed.get("unit")):
                    raise ValueError("subscription quota exhausted or invalid")
                quota = dict(observed)
        return {"accounting_mode": "subscription", "accounting_version": contract.version,
                "subscription_contract": asdict(contract), "quota": quota,
                "usd_applicability": "NOT_APPLICABLE", "usd_state": "UNAVAILABLE"}

    def prepare_request(self, request: Mapping[str, Any], resolved_agent: Any) -> dict[str, Any]:
        # SDK extra_body wins over typed kwargs. Budget and validate that same
        # body, including schemas, instructions and all prior tool results.
        wire = json.loads(json.dumps(dict(request), allow_nan=False))
        extra = wire.pop("extra_body", None)
        if extra is not None:
            if not isinstance(extra, dict):
                raise ValueError("JSON request body required")
            wire.update(extra)
        if resolved_agent.api_mode == "codex_responses":
            if wire.get("context_management"):
                raise ValueError("automatic Responses compaction forbidden in restricted task")
            wire = _sanitize_consumer_codex_request(resolved_agent, wire)
            backend = getattr(resolved_agent, "_is_codex_backend", None)
            self._consumer_codex = callable(backend) and bool(backend())
            if self._consumer_codex:
                # This consumer endpoint has no server-enforced output cap.
                wire.pop("max_output_tokens", None)
            wire["stream"] = True
        return wire

    def physical_output_cap(self, request: Mapping[str, Any], route) -> bool:
        if route.api_mode == "codex_responses" and self._consumer_codex:
            return False
        fields = (("max_output_tokens",) if route.api_mode == "codex_responses"
                  else ("max_tokens", "max_completion_tokens"))
        caps = [request[name] for name in fields if name in request]
        return bool(caps) and all(type(v) is int and 0 < v <= route.max_output_tokens for v in caps)

    @staticmethod
    def _sdk_kwargs(wire: dict[str, Any], create: Callable) -> dict[str, Any]:
        # The final body may contain provider-specific extra_body fields which
        # are not declared SDK kwargs. Preserve those bytes via its merge path.
        parameters = inspect.signature(create).parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            return wire
        extras = {key: value for key, value in wire.items() if key not in parameters}
        if not extras:
            return wire
        return {**{key: value for key, value in wire.items() if key in parameters},
                "extra_body": extras}

    def supports_effort(self, route) -> bool:
        return route.api_mode in {"chat_completions", "codex_responses"}

    def effective_effort(self, request: Mapping[str, Any], route) -> str | None:
        self._requested_tier = _tier(request.get("service_tier"))
        if route.api_mode == "codex_responses":
            reasoning = request.get("reasoning")
            return reasoning.get("effort") if isinstance(reasoning, dict) else None
        top = request.get("reasoning_effort")
        if isinstance(top, str):
            return top
        if isinstance(request.get("reasoning"), dict):
            return request["reasoning"].get("effort")
        body = request.get("extra_body")
        reasoning = body.get("reasoning") if isinstance(body, dict) else None
        return reasoning.get("effort") if isinstance(reasoning, dict) else None

    def client_input_budget(self, request, route):
        self.accounting_contract(route)
        contract = self.subscription_contract
        if self.accounting_mode != "subscription" or contract.input_policy_version != "client-budget-v1":
            raise ValueError("approved subscription client budget required")
        encoded = json.dumps(dict(request), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(encoded) > contract.max_wire_bytes:
            raise ValueError("complete client payload byte ceiling exceeded")
        import hashlib
        self._client_body = json.loads(encoded)
        return {"version": "client-budget-v1", "max_wire_bytes": contract.max_wire_bytes,
                "prepared_payload_bytes": len(encoded), "prepared_payload_sha256": hashlib.sha256(encoded).hexdigest(),
                "token_estimate": contract.token_estimate, "token_estimate_kind": "ESTIMATE",
                "server_input_before_send": "UNKNOWN", "server_output_before_send": "UNKNOWN",
                "request_limit": contract.request_limit, "deadline_seconds": contract.deadline_seconds}

    def input_token_upper_bound(self, request: Mapping[str, Any], route) -> int:
        if (self.accounting_mode == "subscription" and self.subscription_contract is not None
                and self.subscription_contract.input_policy_version == "client-budget-v1"):
            raise ValueError("client token estimate is not a server upper bound")
        # Isolate the approved body from a host estimator that mutates its input.
        encoded = json.dumps(dict(request), sort_keys=True, allow_nan=False)
        bound = self.input_bound(json.loads(encoded), route) if self.input_bound is not None else 0
        if type(bound) is not int or bound <= 0:
            raise ValueError("positive integer whole-request input bound required")
        self._bound_request = (encoded, bound)
        return bound

    def _priced(self, input_tokens: int, output_tokens: int, route) -> float | None:
        if self.accounting_mode == "subscription":
            return None
        quote = self.price
        if (not isinstance(quote, PriceQuote) or quote.version != self.approved_price_version
                or (quote.provider, quote.model) != (route.provider, route.model)
                or _tier(quote.service_tier) != self._requested_tier):
            return None
        return (input_tokens * quote.input_usd_per_million +
                output_tokens * quote.output_usd_per_million) / 1_000_000

    def cost_upper_bound(self, input_tokens: int, output_tokens: int, route) -> float | None:
        return self._priced(input_tokens, output_tokens, route)

    def actual_cost(self, response: Any, route) -> float | None:
        if self.accounting_mode == "subscription":
            return None
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        if self.price is None or _tier(getattr(response, "service_tier", None)) != _tier(self.price.service_tier):
            return None
        if route.api_mode == "codex_responses":
            input_tokens, output_tokens = getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None)
        else:
            input_tokens, output_tokens = getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)
        if type(input_tokens) is not int or type(output_tokens) is not int or min(input_tokens, output_tokens) < 0:
            return None
        return self._priced(input_tokens, output_tokens, route)

    def _measured_usage(self, response: Any, route) -> dict[str, int | float | None]:
        usage = getattr(response, "usage", None)
        codex = route.api_mode == "codex_responses"
        input_tokens = getattr(usage, "input_tokens" if codex else "prompt_tokens", None)
        output_tokens = getattr(usage, "output_tokens" if codex else "completion_tokens", None)
        return {
            "input_tokens": input_tokens if type(input_tokens) is int and input_tokens >= 0 else None,
            "output_tokens": output_tokens if type(output_tokens) is int and output_tokens >= 0 else None,
            "cost_usd": self.actual_cost(response, route),
        }

    def complete(self, request: Mapping[str, Any], resolved_agent: Any) -> Any:
        wire = self.prepare_request(request, resolved_agent)
        route = SimpleNamespace(provider=resolved_agent.provider,
                                model=str(wire.get("model") or getattr(resolved_agent, "model", "")),
                                api_mode=resolved_agent.api_mode)
        self._requested_tier = _tier(wire.get("service_tier"))
        encoded = json.dumps(wire, sort_keys=True, allow_nan=False)
        client_budget = (self.accounting_mode == "subscription" and self.subscription_contract is not None
                         and self.subscription_contract.input_policy_version == "client-budget-v1")
        if client_budget:
            if hasattr(self, "_client_body") and self._client_body != wire:
                raise ValueError("prepared client payload changed after admission")
            self.client_input_budget(wire, route)
            if (self._dispatches >= self.subscription_contract.request_limit or
                    time.monotonic() - self._created_at >= self.subscription_contract.deadline_seconds):
                raise ValueError("client request or deadline budget exhausted")
            bound = None
        else:
            bound = (self._bound_request[1] if self._bound_request and self._bound_request[0] == encoded
                     else self.input_token_upper_bound(wire, route))
        self.accounting_contract(route)
        if self.accounting_mode == "metered" and self.cost_upper_bound(bound, 1, route) is None:
            raise ValueError("host-approved versioned route pricing required before dispatch")
        if client_budget:
            self.deadline = min(self.deadline, self._created_at + self.subscription_contract.deadline_seconds)
        remaining = self.deadline - time.monotonic()
        if remaining <= 0 or self.cancelled():
            raise TimeoutError("reasoning request expired before dispatch")
        client = resolved_agent._create_request_openai_client(reason="memory_reasoning", api_kwargs=wire)
        if client_budget:
            if client.max_retries != 0:
                resolved_agent._close_request_openai_client(client, reason="memory_reasoning_retries_forbidden")
                raise ValueError("client budget requires SDK max_retries=0")
            expected = dict(wire)
            controls = {"timeout", "extra_headers", "extra_query"}
            expected = {k: v for k, v in expected.items() if k not in controls}
            def guard_payload(http_request):
                body = http_request.content
                if (len(body) > self.subscription_contract.max_wire_bytes or
                        json.loads(body) != expected):
                    raise ValueError("SDK client payload changed or exceeded byte ceiling")
                if self._dispatches >= self.subscription_contract.request_limit:
                    raise ValueError("client request budget exhausted")
                self._dispatches += 1
            client._client.event_hooks.setdefault("request", []).append(guard_payload)
        finished = threading.Event()
        aborted = threading.Event()

        def watch() -> None:
            while not finished.is_set():
                remaining = self.deadline - time.monotonic()
                if self.cancelled() or remaining <= 0:
                    aborted.set()
                    resolved_agent._abort_request_openai_client(client, reason="memory_reasoning_cancel")
                    return
                finished.wait(min(0.02, remaining))

        watcher = threading.Thread(target=watch, name="memory-reasoning-request-watch", daemon=True)
        watcher.start()
        stream = None
        try:
            self._raise_if_stopped(aborted)
            wire["timeout"] = max(0.001, self.deadline - time.monotonic())
            if resolved_agent.api_mode == "codex_responses":
                self._raise_if_stopped(aborted)
                stream = client.responses.create(**bypass_sdk_request_transform(
                    self._sdk_kwargs(wire, client.responses.create)))
                completed_usage = None
                completed_tier = None

                def remember_completed(event):
                    nonlocal completed_usage, completed_tier
                    kind = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)
                    if kind == "response.completed":
                        response = event.get("response") if isinstance(event, dict) else getattr(event, "response", None)
                        if isinstance(response, dict):
                            raw = response.get("usage")
                            response = SimpleNamespace(usage=SimpleNamespace(**raw) if isinstance(raw, dict) else None,
                                                       service_tier=response.get("service_tier"))
                        completed_tier = getattr(response, "service_tier", None)
                        completed_usage = self._measured_usage(response, route)

                try:
                    result = _consume_codex_event_stream(
                        stream, model=str(wire["model"]), on_event=remember_completed,
                        interrupt_check=lambda: self._raise_if_stopped(aborted),
                    )
                    if completed_usage is not None:
                        result.service_tier = completed_tier
                except TimeoutError:
                    if completed_usage is not None:
                        raise CompletedUsageInterrupted(completed_usage) from None
                    raise
            else:
                self._raise_if_stopped(aborted)
                result = client.chat.completions.create(**bypass_chat_sdk_request_transform(
                    self._sdk_kwargs(wire, client.chat.completions.create), client))
            try:
                self._raise_if_stopped(aborted)
            except TimeoutError:
                raise CompletedUsageInterrupted(completed_usage if resolved_agent.api_mode == "codex_responses"
                                                and completed_usage is not None else
                                                self._measured_usage(result, route)) from None
            return result
        finally:
            finished.set()
            watcher.join(timeout=0.1)
            try:
                if stream is not None:
                    stream.close()
            finally:
                resolved_agent._close_request_openai_client(client, reason="memory_reasoning_complete")

    def _raise_if_stopped(self, aborted: threading.Event) -> bool:
        if aborted.is_set() or self.cancelled() or time.monotonic() >= self.deadline:
            raise TimeoutError("reasoning request cancelled or deadline exceeded")
        return False
