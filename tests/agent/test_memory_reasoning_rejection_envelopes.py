"""Rejection paths that must not abort a restricted reasoning task.

P7: a document reference outside the host's grants is a rejected, charged read.
    The reader returns its error envelope (status 'error', error_type
    'PermissionDenied') instead of raising, which the runner used to upgrade to
    ``host_read_failed`` and abort the whole task.
P5: the between-turns registry refresh must never rebuild this agent's
    instance-local restricted tool surface. It once emptied it (the placeholder
    toolset resolves to no tools and the five schemas are never registered), so
    the built request carried no ``tools`` array at all and the runner's exact
    surface check rejected every send.
"""

from collections import defaultdict
from types import SimpleNamespace
import json
import logging
import threading
import time

import pytest

from agent.memory_reasoning import (
    BudgetLimits, Evidence, HostHandlers, ReasoningTask, RestrictedReasoningRunner, Route,
)
from agent.memory_reasoning.document_reader import (
    LocalDocumentGrant, ScopedDocumentEvidenceReader,
)
from agent.memory_reasoning.runner import TOOL_NAMES


@pytest.fixture
def fake_route(monkeypatch):
    monkeypatch.setattr(
        "agent.auxiliary_client.resolve_provider_client",
        lambda *args, **kwargs: (SimpleNamespace(api_key="synthetic-test-only",
                                                 base_url="http://localhost/v1"), "test-model"),
    )
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", lambda **kwargs: SimpleNamespace(close=lambda: None))


class Ledger:
    def __init__(self):
        self.values = defaultdict(float)
        self.lock = threading.Lock()

    def reserve(self, task, dimension, amount, ceiling):
        with self.lock:
            key = (task, dimension)
            if self.values[key] + amount > ceiling:
                return False
            self.values[key] += amount
            return True

    def record(self, task, dimension, amount):
        with self.lock:
            self.values[(task, dimension)] += amount

    def begin_attempt(self, task, *, cost_contract=None):
        return 1

    def settle_actual(self, attempt_id, dimension, amount):
        return True

    def snapshot(self, task):
        with self.lock:
            return {dimension: value for (key, dimension), value in self.values.items() if key == task}


def call(name, args, i=1):
    return SimpleNamespace(id=f"call_{i}", type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def response(calls=(), content=None, reasoning=None, finish_reason=None):
    message = SimpleNamespace(content=content, reasoning=reasoning, tool_calls=list(calls))
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message,
                                 finish_reason=finish_reason or ("tool_calls" if calls else "stop"))],
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10),
    )


class RecordingTransport:
    """Records the wire request and answers from a local script (no network)."""

    single_attempt = True

    def __init__(self, script=()):
        self.script = list(script)
        self.requests = []
        self.wire_configs = []
        self.request_keys = []

    def supports_effort(self, route):
        return True

    def effective_effort(self, request, route):
        # The synthetic provider emits no real wire reasoning bytes (its model name is not a
        # reasoning model), so this mirror reports the route's approved effort. The effort
        # half of the runner's predicate is exercised in production; here the observable
        # levers core flips on a continuation are the recorded wire reasoning config
        # (agent._wire_reasoning_config) and the output cap.
        return route.effort

    def input_token_upper_bound(self, request, route):
        return 100

    def cost_upper_bound(self, input_tokens, output_tokens, route):
        return 0.01

    def actual_cost(self, response, route):
        return 0.001

    def complete(self, request, resolved_agent):
        self.requests.append(request)
        self.wire_configs.append(getattr(resolved_agent, "_wire_reasoning_config", "<absent>"))
        self.request_keys.append(sorted(request))
        return self.script.pop(0)


def build_runner(*, script, document_read, transport=None, memory_search=None):
    task = ReasoningTask("task-1", "owner-1", "rev-1", "Host context")
    handlers = HostHandlers(
        memory_search=memory_search or (lambda q, meter: Evidence_memory(meter)),
        exact_read=lambda r, meter: Evidence_memory(meter),
        source_trace=lambda r, meter: Evidence_memory(meter),
        document_read=document_read,
        submit_proposal=lambda submission: "receipt:1",
    )
    return RestrictedReasoningRunner(
        task=task, route=Route("openai", "test-model", "high", max_output_tokens=40),
        limits=BudgetLimits(5, 8, 8, 1000, 200, 60, 0.1),
        handlers=handlers, ledger=Ledger(), transport=transport or RecordingTransport(script),
        cancelled=lambda: False, deadline_monotonic=time.monotonic() + 60,
    )


def Evidence_memory(meter):
    from agent.memory_reasoning import Evidence
    meter()
    return Evidence("memory", "memory:1", "sha256:t", "line:1", "2026-10-01T00:00:00Z",
                    "complete", "bounded exact read", {"text": "x"})


def test_document_out_of_scope_reference_returns_envelope_without_aborting(fake_route, tmp_path):
    """P7: a rejected document read is charged evidence, never a task-killing raise."""
    approved = tmp_path / "approved.txt"
    approved.write_text("alpha\n", encoding="utf-8")
    ref = "document:approved"
    reader = ScopedDocumentEvidenceReader(
        grants={ref: LocalDocumentGrant(root=tmp_path, path=approved)})

    reads = []
    denied = reader.read("document:../outside.txt", lambda: reads.append(1))
    assert (denied.status, denied.content["error_type"]) == ("error", "PermissionDenied")
    assert len(reads) == 1 and denied.underlying_reads == 1
    assert denied.location == "unbound"

    # A hostile/oversized reference is bounded in the envelope, never echoed whole.
    long_reads = []
    bounded = reader.read("document:" + "x" * 5000, lambda: long_reads.append(1))
    assert (bounded.status, bounded.content["error_type"]) == ("error", "PermissionDenied")
    assert len(bounded.exact_reference) == 200
    assert len(long_reads) == 1 and bounded.underlying_reads == 1

    # The approved grant still reads: rejection did not widen or narrow the scope.
    assert reader.read(ref, lambda: None).status == "complete"

    # End to end: the model asking for the out-of-scope document must not end the task.
    transport = RecordingTransport([
        response([call("reason_document_read", {"reference": "document:../outside.txt"})]),
        response(content="continued after the rejected read"),
    ])
    job = build_runner(script=None, document_read=reader.read, transport=transport)
    result = job.run()
    assert result.status != "restricted_failure", result.answer
    assert result.answer == "continued after the rejected read"
    assert len(transport.requests) == 2


def test_turn_start_mcp_refresh_cannot_empty_restricted_tool_surface(fake_route, monkeypatch):
    """P5: the between-turns registry refresh must leave the five tools untouched."""
    import tools.mcp_tool  # noqa: F401  (the gateway imports it; the refresh gate keys on this)
    from tools.mcp_tool_common import _core as mcp_core
    from tools.mcp_tool_discovery import has_registered_mcp_tools

    monkeypatch.setitem(mcp_core._mcp_tool_server_names, "mcp__regression__fake", "regression")
    assert has_registered_mcp_tools() is True, "gateway condition not reproduced"

    transport = RecordingTransport([response(content="surface intact")])
    job = build_runner(script=None, document_read=lambda r, m: Evidence_memory(m), transport=transport)
    result = job.run()
    assert result.status != "restricted_failure", result.answer
    assert len(transport.requests) == 1, "the restricted request was never dispatched"
    sent = transport.requests[0]["tools"]
    assert [tool["function"]["name"] for tool in sent] == list(TOOL_NAMES)


def test_malformed_arguments_are_rejected_without_aborting_the_task(fake_route, caplog):
    """P8: a wrong-shaped call for an ALLOWED tool is rejected back to the model.

    Nothing is dispatched, no logical tool is charged, the loop stays inside its
    budget, and a name outside the surface still stops the task.
    """
    seen = []

    def search(query, meter):
        meter()
        seen.append(query)
        return Evidence("memory", "memory:1", "sha256:t", "line:1", "2026-10-01T00:00:00Z",
                        "complete", "bounded exact read", {"text": query})

    transport = RecordingTransport([
        response([call("reason_memory_search", {"query": "first", "unexpected_key": 1})]),
        response([call("reason_memory_search", {"query": "second"})]),
        response(content="recovered after the rejected call"),
    ])
    job = build_runner(script=None, document_read=lambda r, m: Evidence_memory(m),
                       transport=transport, memory_search=search)
    with caplog.at_level(logging.WARNING, logger="agent.memory_reasoning.runner"):
        result = job.run()

    assert result.status != "restricted_failure", result.answer
    assert result.answer == "recovered after the rejected call"
    assert seen == ["second"], "the malformed call must never reach the host handler"
    assert result.usage.get("logical_tools", 0) == 1
    assert result.usage.get("underlying_reads", 0) == 1
    assert len(transport.requests) == 3
    # Bounded structural log: tool name and argument keys, never the argument values.
    assert "rejected (malformed_arguments)" in caplog.text
    assert "unexpected_key" in caplog.text and "first" not in caplog.text


def test_foreign_tool_name_still_aborts_the_task(fake_route):
    """P8: a name OUTSIDE the restricted surface stays fatal — the surface invariant."""
    transport = RecordingTransport([response([call("terminal", {"command": "echo unsafe"})])])
    job = build_runner(script=None, document_read=lambda r, m: Evidence_memory(m),
                       transport=transport)
    result = job.run()
    assert result.status == "restricted_failure"
    assert "forbidden or malformed tool call" in result.answer
    assert result.usage.get("logical_tools", 0) == 0
    assert len(transport.requests) == 1


def test_compaction_boundary_refresh_honours_the_surface_opt_out(monkeypatch):
    """P8 (defence in depth): the compaction-boundary rebuild obeys the same opt-out."""
    import agent.conversation_compression as cc

    calls = []

    def record(*args, **kwargs):
        calls.append(args)
        return set()

    monkeypatch.setattr("tools.mcp_tool_agent.refresh_agent_mcp_tools", record)

    assert cc._refresh_agent_tool_definitions(SimpleNamespace(_skip_mcp_refresh=True)) is False
    assert calls == [], "an opted-out instance must not be rebuilt from the registry"

    assert cc._refresh_agent_tool_definitions(SimpleNamespace(_skip_mcp_refresh=False)) is False
    assert len(calls) == 1, "ordinary agents keep the compaction-boundary refresh"


def test_plain_spin_still_stops_at_two_strikes(fake_route):
    """P14 (invariant): evidence-less spinning that never tried to submit still stops at 2."""
    transport = RecordingTransport([
        response([call("reason_memory_search", {"query": "project database"})]),
        response([call("reason_exact_read", {"reference": "source:not-this-task"})]),
        response([call("reason_exact_read", {"reference": "source:not-this-task"}, 2)]),
        response(content="unused"),
    ])
    job = build_runner(script=None, document_read=lambda r, m: Evidence_memory(m),
                       transport=transport)
    result = job.run()
    # The runner maps any ToolBudgetExceeded to "budget_exhausted"; the worker then reports
    # it as BUDGET_EXHAUSTED, which is why P12 read the guard as a budget stop.
    assert result.status == "budget_exhausted"
    assert "no_progress" in result.answer
    assert len(transport.requests) == 3, "the third send must be the last one dispatched"
    assert result.proposal_receipt is None


def test_rejected_proposal_attempt_earns_one_bounded_correction(fake_route):
    """P14: when the SECOND consecutive no-evidence send carried a rejected submission
    attempt, the guard allows one more send — and a valid submission in it still forms the
    proposal. This mirrors the live sequence (strike 1 from a bare no-evidence send, strike 2
    from the send whose proposal was rejected for shape)."""
    malformed = call("reason_proposal_submit", {"proposal": {"action": "UPDATE"}})
    # Mirrors the live signature: the arguments PARSE but are not an object, so the call is
    # rejected by shape without the loop treating the action as cut off.
    malformed.function.arguments = "null"
    # The runner only accepts a proposal whose evidence_refs it can resolve against what it
    # actually read (the search below serves memory:1).
    submission = {"action": "UPDATE", "evidence_refs": ["memory:1"]}
    transport = RecordingTransport([
        response([call("reason_memory_search", {"query": "project database"})]),
        response([call("reason_exact_read", {"reference": "source:not-this-task"})]),
        response([malformed]),
        response([call("reason_proposal_submit", {"proposal": submission})]),
        response(content="done"),
    ])
    job = build_runner(script=None, document_read=lambda r, m: Evidence_memory(m),
                       transport=transport)
    result = job.run()

    assert result.status != "restricted_failure", result.answer
    assert result.proposal_receipt, result.status
    assert result.status == "proposal_submitted"
    assert len(transport.requests) >= 4, "the rejected attempt must not end the run"


def test_unrecoverable_submit_failure_keeps_the_host_reason_code_and_aborts(fake_route):
    """P15/P17 invariant: a submit failure that is NOT a proposal-shape rejection still keeps
    the host's code and still ends the round (the recoverable family is listed in the runner)."""
    import dataclasses
    from agent.memory_reasoning import runner as runner_mod

    # The helper is the whole contract: keep the code, never leak raw text.
    assert runner_mod._host_submit_detail(ValueError('read_budget_exhausted')) == 'read_budget_exhausted'
    assert runner_mod._host_submit_detail(ValueError()) == 'ValueError'
    assert runner_mod._host_submit_detail(RuntimeError('bad\nsecond line')) == 'bad'

    def refuse(submission):
        raise ValueError('read_budget_exhausted')

    submission = {"action": "UPDATE", "evidence_refs": ["memory:1"]}
    transport = RecordingTransport([
        response([call("reason_memory_search", {"query": "project database"})]),
        response([call("reason_proposal_submit", {"proposal": submission})]),
        response(content="done"),
    ])
    job = build_runner(script=None, document_read=lambda r, m: Evidence_memory(m),
                       transport=transport)
    job.handlers = dataclasses.replace(job.handlers, submit_proposal=refuse)
    result = job.run()

    assert job._stop_reason == 'host_submit_failed:read_budget_exhausted'
    assert result.status == "restricted_failure", result.status
    assert "read_budget_exhausted" in result.answer, result.answer


def test_structured_proposal_shape_rejection_returns_the_host_reason_and_continues(fake_route):
    """P17: a host rejection of the proposal's SHAPE does not end the round — the model gets the
    host's own code back and may resubmit inside the existing budgets (P8/P14 family)."""
    import dataclasses
    from agent.memory_reasoning import runner as runner_mod
    seen = []

    # Only the STRUCTURED_PROPOSAL_* family is a recoverable shape slip; every other host
    # rejection (budget, source, planning) stays a hard stop.
    assert runner_mod._proposal_shape_rejection('STRUCTURED_PROPOSAL_ITEM_INVALID')
    assert runner_mod._proposal_shape_rejection('STRUCTURED_PROPOSAL_REFS_OR_ITEMS_INVALID')
    assert not runner_mod._proposal_shape_rejection('read_budget_exhausted')
    assert not runner_mod._proposal_shape_rejection('PLANNING_DUPLICATE_EVIDENCE')
    assert not runner_mod._proposal_shape_rejection('TASK_SOURCE_INVALID')

    def refuse_then_accept(submission):
        seen.append(submission.proposal_json)
        if len(seen) == 1:
            raise ValueError('STRUCTURED_PROPOSAL_REFS_OR_ITEMS_INVALID')
        return 'receipt:shape-fixed'

    bad = {"action": "UPDATE", "evidence_refs": ["memory:1"], "note": "wrong shape"}
    good = {"action": "UPDATE", "evidence_refs": ["memory:1"]}
    transport = RecordingTransport([
        response([call("reason_memory_search", {"query": "project database"})]),
        response([call("reason_proposal_submit", {"proposal": bad})]),
        response([call("reason_proposal_submit", {"proposal": good})]),
        response(content="done"),
    ])
    job = build_runner(script=None, document_read=lambda r, m: Evidence_memory(m),
                       transport=transport)
    job.handlers = dataclasses.replace(job.handlers, submit_proposal=refuse_then_accept)
    result = job.run()

    assert job._stop_reason is None, job._stop_reason
    assert result.status == "proposal_submitted", (result.status, result.answer)
    assert result.proposal_receipt == 'receipt:shape-fixed'
    # The corrected JSON reached the host, which is only possible if the frozen submission was
    # rolled back (otherwise the host answers a different revision with a conflict).
    assert len(seen) == 2, seen
    assert seen[0] != seen[1]
    # The model saw the host's own code in the tool result of the rejected submission.
    assert 'STRUCTURED_PROPOSAL_REFS_OR_ITEMS_INVALID' in json.dumps(
        transport.requests[-1], ensure_ascii=False)


def test_thinking_only_truncation_continuation_keeps_the_approved_wire_profile(fake_route):
    """P11: a thinking-only truncation continues on the SAME approved effort and cap.

    Core's continuation heuristics drop reasoning (and boost the output cap) when thinking
    ate the budget; for a host-approved restricted job that would change the approved wire
    profile, which the runner refuses. The continuation must re-issue identically.
    """
    transport = RecordingTransport([
        response(content=None, reasoning="thinking that consumed the whole cap",
                 finish_reason="length"),
        response(content="recovered after the continuation"),
    ])
    job = build_runner(script=None, document_read=lambda r, m: Evidence_memory(m),
                       transport=transport)
    result = job.run()

    assert result.status != "restricted_failure", result.answer
    assert result.answer == "recovered after the continuation"
    assert len(transport.requests) == 2, "the truncated response must be continued, not aborted"
    approved = {"enabled": True, "effort": job.route.effort}
    assert transport.wire_configs == [approved, approved], transport.request_keys
    continuation = transport.requests[1]
    assert continuation.get("max_tokens") == job.route.max_output_tokens
