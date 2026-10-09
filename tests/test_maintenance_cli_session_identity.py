"""CLI source identity stays durable when SessionDB has no platform session key."""

from contextlib import closing
from dataclasses import replace
import sqlite3

import pytest

import hermes_maintenance_source as sources
from hermes_state import SessionDB


def _append(db, session_id, receipt, *, batch=False, **fields):
    row = {"role": "user", "content": receipt.extraction, "_maintenance_source": receipt, **fields}
    if batch:
        db.append_messages_batch(session_id, [row])
        return row["_row_id"]
    return db.append_message(session_id, **row)


def _stored_source(home, source_id):
    with closing(sqlite3.connect(home / "state.db")) as conn:
        conn.row_factory = sqlite3.Row
        return dict(conn.execute("SELECT * FROM maintenance_sources_v1 WHERE source_id=?",
                                 (source_id,)).fetchone())


@pytest.mark.parametrize("session_key,legacy", [(None, False), ("", False),
                                                ("cli:explicit", False), ("", True)])
@pytest.mark.parametrize("mode", ["single", "batch", "control"])
def test_cli_source_identity_roundtrip(tmp_path, session_key, legacy, mode):
    home = tmp_path / "home"
    home.mkdir()
    db = SessionDB(db_path=home / "state.db")
    db.create_session("local-session", "cli", session_key=session_key)
    if mode == "control":
        receipt = sources.mint_local_control(
            "remember this", home=home, session_id="local-session", kind="busy_queue")
        source_id = sources.commit_local_control_source(home, "local-session", receipt)
    else:
        receipt = sources.mint_local_input("remember this", home=home, session_id="local-session")
        _append(db, "local-session", receipt, batch=mode == "batch")
        source_id = sources.pending_sources(home)[0]
    if legacy and mode != "control":
        # Pre-fix CLI text rows could store/sign an empty key. Recreate that persisted format.
        with closing(sqlite3.connect(home / "state.db")) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("UPDATE maintenance_sources_v1 SET session_key='' WHERE source_id=?", (source_id,))
            old = conn.execute("SELECT * FROM maintenance_sources_v1 WHERE source_id=?", (source_id,)).fetchone()
            conn.execute("UPDATE maintenance_sources_v1 SET record_digest=? WHERE source_id=?",
                         (sources._record_digest(old), source_id))
            conn.commit()
    stored = _stored_source(home, source_id)
    db.close()
    db = SessionDB(db_path=home / "state.db")
    source = sources.read_committed_source(home, source_id)
    assert source is not None
    assert source["session_key"] == ("" if legacy and mode != "control" else session_key or "local-session")
    assert source["session_id"] == "local-session"
    replay = sources.rehydrate_source_receipt(home, source_id)
    assert replay.session_key == (session_key or "local-session")
    assert sources.admit_reused_local_source(replay, home=home, session_id="local-session") is replay
    assert sources.admit_reused_local_source(replay, home=home, session_id="other") is None
    assert sources.admit_reused_local_source(replay, home=tmp_path, session_id="local-session") is None
    row_id = _append(db, "local-session", replay, batch=True)
    assert sources.source_for_session_row(home, "local-session", row_id) == source_id
    assert sources.committed_source_for_receipt(home, "local-session", replay) == source
    assert sources.pending_sources(home) == [source_id]
    if mode == "control":
        assert sources.commit_local_control_source(home, "local-session", replay) == source_id
        for fields in ({"session_key": "forged"}, {"intended_session_id": "other"},
                       {"home": str(tmp_path)}, {"seal": object()},
                       {"namespace": "forged"}, {"actor": "forged"}):
            forged = replace(replay, **fields)
            assert sources.commit_local_control_source(home, "local-session", forged) is None
            bad_row = _append(db, "local-session", forged)
            assert sources.source_for_session_row(home, "local-session", bad_row) is None
    else:
        db.create_session("clone", "cli", session_key="cli:clone")
        clone_row = _append(db, "clone", replay)
        assert sources.source_for_session_row(home, "clone", clone_row) == source_id
        assert sources.read_committed_source(home, source_id)["session_id"] == "local-session"
    assert _stored_source(home, source_id) == stored  # Reading/replay never re-signs history.
    edited = sources.revise_original_input(replay, "edited fact")
    if mode == "control":
        assert sources.commit_local_control_source(home, "local-session", edited)
    else:
        _append(db, "local-session", edited)
    revised = sources.committed_source_for_receipt(home, "local-session", edited)
    assert revised["session_key"] == (session_key or "local-session")
    assert revised["source_id"] != source_id
    assert len(sources.pending_sources(home)) == 2
    with closing(sqlite3.connect(home / "state.db")) as conn:
        assert conn.execute("SELECT session_key FROM sessions WHERE id='local-session'").fetchone()[0] == session_key
        conn.execute("UPDATE sessions SET session_key='changed' WHERE id='local-session'")
        conn.commit()
    assert sources.read_committed_source(home, source_id) is None
    assert sources.rehydrate_source_receipt(home, source_id) is None
    db.close()


@pytest.mark.parametrize("session_key", [None, "", "cli:explicit"])
@pytest.mark.parametrize("batch", [False, True], ids=["single", "batch"])
def test_cli_identity_keeps_scope_and_transaction_boundaries(tmp_path, monkeypatch, session_key, batch):
    home = tmp_path / "home"
    home.mkdir()
    db = SessionDB(db_path=home / "state.db")
    db.create_session("local-session", "cli", session_key=session_key)
    receipt = sources.mint_local_input("original fact", home=home, session_id="local-session")
    forged_fields = ({"seal": object()}, {"home": str(tmp_path)},
                     {"intended_session_id": "other"}, {"session_key": "forged"},
                     {"session_key": 42}, {"namespace": "forged"}, {"actor": "forged"},
                     {"origin": "model"}, {"audience": "group"})
    for fields in forged_fields:
        _append(db, "local-session", replace(receipt, **fields), batch=batch)
        assert sources.pending_sources(home) == []
    for fields in ({"role": "assistant"}, {"content": "different"}, {"display_kind": "hidden"},
                   {"_compressed_summary": True}, {"platform_message_id": "invented"}):
        _append(db, "local-session", receipt, batch=batch, **fields)
        assert sources.pending_sources(home) == []
    db.create_session("remote-session", "qqbot", session_key=None)
    _append(db, "remote-session", receipt, batch=batch)
    assert sources.pending_sources(home) == []
    row_id = _append(db, "local-session", receipt, batch=batch)
    source_id = sources.source_for_session_row(home, "local-session", row_id)
    assert source_id is not None
    replay = sources.rehydrate_source_receipt(home, source_id)
    for fields in ({"intended_session_id": "other"}, {"session_key": "forged"}, {"session_key": 42}):
        bad_row = _append(db, "local-session", replace(replay, **fields), batch=batch)
        assert sources.source_for_session_row(home, "local-session", bad_row) is None
    with closing(sqlite3.connect(home / "state.db")) as conn:
        before = conn.execute("SELECT count(*) FROM messages").fetchone()[0]

    def fail_after_source(*args, **kwargs):
        conn = args[0]
        assert conn.execute("SELECT count(*) FROM maintenance_sources_v1").fetchone()[0] == 2
        assert conn.execute("SELECT count(*) FROM maintenance_outbox_v1").fetchone()[0] == 2
        raise RuntimeError("downstream persistence failed")

    later = sources.mint_local_input("later fact", home=home, session_id="local-session")
    with monkeypatch.context() as patch:
        patch.setattr(db, "_bump_session_counters", fail_after_source)
        with pytest.raises(RuntimeError, match="downstream persistence failed"):
            _append(db, "local-session", later, batch=batch)
    with closing(sqlite3.connect(home / "state.db")) as conn:
        assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == before
        assert conn.execute("SELECT count(*) FROM maintenance_sources_v1").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM maintenance_outbox_v1").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM maintenance_source_projections_v1").fetchone()[0] == 1
    assert sources.pending_sources(home) == [source_id]
    _append(db, "local-session", later, batch=batch)
    assert sources.committed_source_for_receipt(home, "local-session", later)
    assert len(sources.pending_sources(home)) == 2
    db.close()
