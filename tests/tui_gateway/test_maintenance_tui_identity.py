"""Read identity requires original native-stdio ingress and a durable session incarnation."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from hermes_maintenance_source import pending_sources
from hermes_maintenance_tui import (
    TuiSessionReceipt, read_trusted_tui_session, revoke_trusted_tui_session,
    take_tui_session_receipt,
)
from hermes_state import SessionDB
from tui_gateway import entry, server
from tui_gateway.transport import StdioTransport


@pytest.fixture
def native_sessions(tmp_path, monkeypatch):
    from agent.secret_scope import is_multiplex_active, set_multiplex_active

    homes = [tmp_path / name for name in ("a", "b")]
    for home in homes:
        home.mkdir()
        (home / "config.yaml").write_text("model: {default: test-model}\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(homes[0]))
    monkeypatch.setattr(server, "_hermes_home", homes[0])
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_schedule_row_git_meta", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_a: None)
    monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *_a: False)
    monkeypatch.setattr(server, "_emit", lambda *_a, **_k: None)
    worker_proofs = []

    def worker(_rid, _sid, session, text, *_args):
        worker_proofs.append(read_trusted_tui_session(session["profile_home"], session["session_key"]))
        assert take_tui_session_receipt(
            _rid, {"session_id": _sid, "text": text}, session,
            home=session["profile_home"], session_id=session["session_key"], text=text) is None
        session["running"] = False

    monkeypatch.setattr(server, "_run_after_agent_ready", worker)
    dbs = [SessionDB(db_path=home / "state.db") for home in homes]
    multiplex = is_multiplex_active()
    set_multiplex_active(True)
    try:
        records = []
        for index, (home, db) in enumerate(zip(homes, dbs)):
            # Same durable id in two real profile DBs must never alias the receipt.
            key, sid = "same-session", f"ui-{index}"
            db.create_session(key, "tui", session_key="native-local")
            db.append_message(key, "user", "legacy imported history")
            with server._session_profile_runtime_scope({"profile_home": str(home)}):
                session = server._deferred_session_record(
                    key, cols=80, cwd=str(tmp_path), history=[], lease=None, profile_home=home)
            server._sessions[sid] = session
            records.append((home, db, sid, session))
        yield records, worker_proofs
    finally:
        set_multiplex_active(multiplex)
        for session in server._sessions.values():
            if thread := session.get("_run_thread"):
                thread.join(timeout=5)
            server._release_active_session_slot(session)
        for db in dbs:
            db.close()


def _submit(record, **params):
    _home, _db, sid, session = record
    request = {"id": "human-input", "method": "prompt.submit",
               "params": {"session_id": sid, "text": "my original local prompt", **params}}
    response = entry._dispatch_stdio_request(request)
    if thread := session.get("_run_thread"):
        thread.join(timeout=5)
        assert not thread.is_alive()
    return response


@pytest.mark.parametrize("queued", [False, True, "mixed"])
def test_native_identity_is_durable_profile_scoped_read_only(native_sessions, monkeypatch, queued):
    import hermes_maintenance_tui as identity

    records, worker_proofs = native_sessions
    first, second = records
    proofs = []
    issued = []
    bind = identity.bind_tui_session_receipt

    def capture(*args):
        issued.append(args[-1])
        return bind(*args)

    monkeypatch.setattr(identity, "bind_tui_session_receipt", capture)
    for record in (first, second, first):
        home, db, sid, session = record
        key = session["session_key"]
        if not proofs or record is second:
            assert read_trusted_tui_session(home, key) is None  # labels and old user rows are insufficient
        session["running"] = bool(queued)
        if not proofs and queued == "mixed":
            response = server.dispatch({"id": "internal-input", "method": "prompt.submit",
                "params": {"session_id": sid, "text": "internal queued input", "queued": True}})
            assert response["result"]["status"] == "queued"
            assert read_trusted_tui_session(home, key) is None
        response = _submit(record, queued=bool(queued))
        assert response["result"]["status"] == ("queued" if queued else "streaming"), response
        proof = read_trusted_tui_session(home, key)
        assert proof and proof["capability"] == "read" and proof["platform"] == "tui"
        assert proof["actor"] == "local-user" and proof["home"] == str(home.resolve())
        assert proof["session_key"] == db.get_session(key)["session_key"]
        assert proof["session_started_at"] == db.get_session(key)["started_at"]
        if not proofs:
            receipt = issued[-1]
            with pytest.raises(FrozenInstanceError):
                receipt.home = str(second[0])
            second[1].append_message(key, "user", receipt.text, _maintenance_source=receipt)
            assert read_trusted_tui_session(second[0], key) is None
            db.create_session("unrelated", "tui")
            db.append_message("unrelated", "user", receipt.text, _maintenance_source=receipt)
            assert read_trusted_tui_session(home, "unrelated") is None
            if queued == "mixed":
                assert session["queued_prompt"]["text"] == "internal queued input"
                assert session["queued_prompts"][0]["text"] == receipt.text
                assert db._conn.execute("SELECT content FROM messages WHERE id=?",
                                        (proof["message_row_id"],)).fetchone()[0] == receipt.text
        assert pending_sources(home) == []
        assert db._conn.execute("SELECT name FROM sqlite_master WHERE name IN "
                                "('maintenance_sources_v1','maintenance_outbox_v1')").fetchall() == []
        # A fresh connection verifies the durable proof, independent of process memory.
        fresh = SessionDB(db_path=home / "state.db")
        try:
            assert read_trusted_tui_session(home, key) == proof
        finally:
            fresh.close()
        if queued:
            session["running"] = False
            monkeypatch.setattr(server, "_run_prompt_submit", lambda *_a, **_k: None)
            while session.get("queued_prompt"):
                assert server._drain_queued_prompt("drain", sid, session)
                session["running"] = False
            assert read_trusted_tui_session(home, key) == proof
        proofs.append(proof)
    assert proofs[0] == proofs[2] and proofs[0]["receipt_id"] != proofs[1]["receipt_id"]
    if not queued:
        assert worker_proofs == proofs
    home, db, _sid, session = first
    proof = proofs[0]
    # Expansion changes the model projection, not the local identity or source enrollment.
    db.set_user_message_content(session["session_key"], proof["message_row_id"], "expanded model input")
    assert read_trusted_tui_session(home, session["session_key"]) == proof
    # Compression/retry cannot carry authority into another session incarnation.
    db.create_session("compression-child", "tui", parent_session_id=session["session_key"])
    db.append_message("compression-child", "user", "my original local prompt", _compressed_summary=True)
    assert read_trusted_tui_session(home, "compression-child") is None
    assert not revoke_trusted_tui_session(home, session["session_key"], proofs[1]["receipt_id"])
    assert revoke_trusted_tui_session(home, session["session_key"], proof["receipt_id"])
    session["running"] = False
    assert "result" in _submit(first)
    assert read_trusted_tui_session(home, session["session_key"]) is None  # revocation is sticky


@pytest.mark.parametrize("untrusted", [
    "direct-handler", "shared-dispatch", "foreign-transport", "desktop", "bot-author", "hidden",
    "dict-receipt", "unsealed-receipt", "system-row", "auto-continue", "row-identity", "row-kind",
    "session-incarnation", "proof-tamper", "deleted-row", "rollback",
    "HERMES_TUI_DASHBOARD", "HERMES_PTY_HOST", "HERMES_DESKTOP", "HERMES_DESKTOP_TERMINAL",
    "steer", "redirect",
])
def test_untrusted_input_or_changed_binding_never_grants_read(native_sessions, monkeypatch, untrusted):
    records, _ = native_sessions
    record = records[0]
    home, db, sid, session = record
    key, text = session["session_key"], "my original local prompt"
    req = {"id": "human-input", "method": "prompt.submit", "params": {"session_id": sid, "text": text}}
    if untrusted.startswith("HERMES_"):
        monkeypatch.setenv(untrusted, "dashboard" if untrusted == "HERMES_PTY_HOST" else "1")
        _submit(record)
    elif untrusted in ("direct-handler", "shared-dispatch"):
        (server.handle_request if untrusted == "direct-handler" else server.dispatch)(req)
    elif untrusted == "foreign-transport":
        session["transport"] = StdioTransport(lambda: None, threading.Lock())
        _submit(record)
    elif untrusted == "desktop":
        session["source"] = "desktop"
        _submit(record)
    elif untrusted == "bot-author":
        assert "error" in _submit(record, _turn_author={"id": "bot"})
    elif untrusted == "hidden":
        _submit(record, display_kind="hidden")
    elif untrusted in ("steer", "redirect"):
        monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer" if untrusted == "steer" else "interrupt")
        session["running"] = True
        session["agent"] = SimpleNamespace(steer=lambda text: True, redirect=lambda text: True,
                                            _supports_active_turn_redirect=True)
        assert _submit(record)["result"]["status"] == ("steered" if untrusted == "steer" else "redirected")
    elif untrusted in ("dict-receipt", "unsealed-receipt"):
        receipt = TuiSessionReceipt(str(home), key, text)
        with pytest.raises(FrozenInstanceError):
            receipt.session_id = "another-session"
        db.append_message(key, "user", text, _maintenance_source=(
            {"actor": "local-user", "platform": "tui"} if untrusted == "dict-receipt" else receipt))
    elif untrusted in ("system-row", "auto-continue"):
        server._persist_submit_user_row(session, text, untrusted)
    elif untrusted == "rollback":
        import hermes_maintenance_tui as identity
        bind = identity.bind_tui_session_receipt

        def fail_after_bind(*args):
            bind(*args)
            raise RuntimeError("rollback receipt with the user row")

        monkeypatch.setattr(identity, "bind_tui_session_receipt", fail_after_bind)
        _submit(record)
        assert db._conn.execute("SELECT count(*) FROM messages WHERE session_id=?", (key,)).fetchone()[0] == 1
    else:
        _submit(record)
        proof = read_trusted_tui_session(home, key)
        assert proof
        if untrusted == "session-incarnation":
            db._conn.execute("UPDATE sessions SET started_at=started_at+1 WHERE id=?", (key,))
        elif untrusted == "proof-tamper":
            db._conn.execute("UPDATE maintenance_tui_sessions_v1 SET actor='foreign-user' WHERE session_id=?", (key,))
        elif untrusted == "deleted-row":
            db._conn.execute("DELETE FROM messages WHERE id=?", (proof["message_row_id"],))
        else:
            column, value = ("message_uid", "replaced") if untrusted == "row-identity" else ("display_kind", "auto_continue")
            db._conn.execute(f"UPDATE messages SET {column}=? WHERE id=?", (value, proof["message_row_id"]))
        db._conn.commit()
    if thread := session.get("_run_thread"):
        thread.join(timeout=5)
    assert read_trusted_tui_session(home, key) is None
    assert read_trusted_tui_session(records[1][0], key) is None
    assert pending_sources(home) == []
