"""Tool failure belongs to the result envelope, not historical business data."""

import copy
import json
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.display import _detect_tool_failure
from agent.tool_guardrails import classify_tool_failure


@pytest.mark.parametrize("tool,payload,failed", [
    ("task_list", {"tasks": [{"status": "FAILED", "error": "old failure"}]}, False),
    ("task_list", {"padding": "x" * 600, "tasks": [{"status": "FAILED"}]}, False),
    ("task_list", {"success": True, "message": "failed", "data": {"success": False}}, False),
    ("task_list", {"failed": 12, "status": {"error": "historical"}}, False),
    ("task_list", [{"status": "FAILED", "error": "old failure"}], False),
    ("task_list", {"error": None, "message": "ok"}, False),
    ("task_list", {"error": ""}, False),
    ("task_list", {"error": False}, False),
    ("task_list", {"error": []}, False),
    ("task_list", {"error": {}}, False),
    ("task_list", {}, False),
    ("task_list", [], False),
    ("task_list", None, False),
    ("task_list", False, False),
    ("task_list", "failed", False),
    ("task_list", "error", False),
    ("task_list", {"success": False}, False),
    ("read_file", {"success": False, "note": "Not a regular file; no read was attempted."}, False),
    ("memory", {"success": False, "error": None}, False),
    ("task_list", {"success": False, "message": "unavailable"}, True),
    ("task_list", {"success": True, "error": "unavailable"}, True),
    ("task_list", {"error": {"code": "unavailable"}}, True),
    ("task_list", {"status": "FAILED"}, True),
    ("task_list", {"status": "Error", "message": "unavailable"}, True),
    ("task_list", {"padding": "x" * 600, "status": "failed"}, True),
    ("task_list", {"padding": "x" * 600, "error": "unavailable"}, True),
    ("terminal", {"exit_code": 0, "status": "failed", "error": "output"}, False),
    ("terminal", {"exit_code": 2}, True),
    ("terminal", {"error": "no exit code"}, False),
    ("terminal", {"user_summary": "Approval was denied", "exit_code": 0}, True),
    ("memory", {"success": False, "error": "would exceed the limit"}, True),
    ("read_file", {"guardrail_refusal": True, "error": "do not retry"}, False),
    ("read_file", {"guardrail_refusal": "yes", "error": "unavailable"}, True),
    ("write_file", {"bytes_written": 10, "output": {"error": "old"}}, False),
    ("patch", {"success": True, "diff": {"status": "failed"}}, False),
    ("browser_exec", {"_multimodal": True, "content": [{"text": "failed"}]}, False),
    ("task_list", 'Error executing tool: unavailable', True),
    ("task_list", 'partial {"error": unavailable', True),
    ("task_list", 'partial {"status": "FAILED"', True),
    ("task_list", 'ordinary successful response', False),
    ("task_list", 'x' * 600 + '"error"', False),
])
def test_result_envelope_classification_and_representation(tool, payload, failed):
    # Plain text keeps the bounded legacy heuristic; JSON scalar strings are data.
    variants = [payload, json.dumps(payload)] if not isinstance(payload, str) or payload in {"failed", "error"} else [payload]
    for result in variants:
        original = copy.deepcopy(result)
        for classify in (_detect_tool_failure, classify_tool_failure):
            actual, suffix = classify(tool, result)
            assert actual is failed
            assert bool(suffix) is failed
        assert result == original


@pytest.mark.parametrize("path", ["sequential", "concurrent"])
@pytest.mark.parametrize("failed", [False, True])
def test_result_classification_reaches_runtime_consumers(path, failed):
    from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController
    from gateway.run_turn_runner import TurnRunner
    from model_tools import _tool_result_observer_fields
    from tests.agent.test_tool_call_guardrail_runtime import _make_agent, _mock_tool_call

    tool = "task_list"
    payload = {"tasks": [{"status": "FAILED", "error": "historical"}]}
    if failed:
        payload["error"] = "unavailable"
    result = json.dumps(payload)
    agent = _make_agent(tool)
    observations = []

    def observe(name, args, value, **kwargs):
        observations.append((name, value, kwargs["failed"]))
        return value

    agent._append_guardrail_observation = observe
    agent._invoke_tool = MagicMock(return_value=result)
    message = SimpleNamespace(content="", tool_calls=[_mock_tool_call(tool)])
    messages = []
    with patch("model_tools.handle_function_call", return_value=result):
        getattr(agent, "_execute_tool_calls_" + path)(message, messages, "task")
    assert observations == [(tool, result, failed)]
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["content"] == result
    assert _tool_result_observer_fields(tool, result)[0] == ("error" if failed else "ok")

    queue = Queue()
    ctx = SimpleNamespace(progress_queue=queue, _run_still_current=lambda: True, agent_holder=[])
    TurnRunner(None, ctx).native_tool_complete_callback("call", tool, {}, result)
    assert queue.get_nowait()["is_error"] is failed

    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=True))
    for page in range(10):
        controller.after_call(tool, {"page": page}, result)
    assert (controller.halt_decision is not None) is failed
