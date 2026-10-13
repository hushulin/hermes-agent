"""Native stdio TUI session identity for read scope, never maintenance intake.

The stdio reader is the local-operator boundary. Shared backend methods, source
labels, transcript imports and model turns cannot create this proof. It records
control of one session incarnation, not authorship of transformed model input.
"""

from contextlib import closing, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from uuid import uuid4


CONTRACT = "tui-local-session-v1"
_SEAL = object()
_INGRESS = ContextVar("maintenance_tui_stdio_ingress", default=None)
_FIELDS = ("receipt_id", "home", "session_id", "session_key", "session_started_at",
           "actor", "message_row_id", "message_uid", "received_at")


@dataclass(frozen=True)
class _StdioPrompt:
    request_id: object
    session_id: str
    text: str
    transport: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class TuiSessionReceipt:
    home: str
    session_id: str
    text: str
    receipt_id: str = field(default_factory=lambda: uuid4().hex)
    received_at: float = field(default_factory=time.time)
    seal: object = field(default=None, repr=False, compare=False)


@contextmanager
def stdio_prompt_request(request, transport):
    """Entry-reader-only admission; copying the dispatch context preserves its peer.

    Hidden, replay/edit, hosted and relay submits are not original local input.
    No wire field can opt into this context.
    """
    params = request.get("params") if isinstance(request, dict) else None
    prompt = None
    # Dashboard PTY fallback and the Desktop terminal also spawn this stdio
    # backend. Their server-set launch markers can only DENY local identity;
    # a session source string or client field can never opt into it.
    hosted = any(os.environ.get(key) for key in (
        "HERMES_TUI_DASHBOARD", "HERMES_PTY_HOST", "HERMES_DESKTOP", "HERMES_DESKTOP_TERMINAL"))
    if (isinstance(params, dict) and request.get("method") == "prompt.submit"
            and isinstance(params.get("session_id"), str) and params["session_id"]
            and isinstance(params.get("text"), str) and params["text"].strip()
            and transport is not None and not hosted
            and not any(params.get(key) is not None for key in (
                "display_kind", "surface", "voice_context", "_turn_author", "_hosted_task",
                "_hosted_terminal_callback", "truncate_before_user_ordinal",
                "truncate_before_row_id", "truncate_before_message_id"))):
        prompt = _StdioPrompt(request.get("id"), params["session_id"], params["text"], transport)
    token = _INGRESS.set(prompt)
    try:
        yield
    finally:
        _INGRESS.reset(token)


def take_tui_session_receipt(request_id, params, session, *, home, session_id, text):
    """Consume one original ingress before the RPC reattaches any transport.

    The receipt is passed explicitly to the accept-time writer, never installed
    on the session or inherited by an agent, queued drain or auto-continue.
    """
    from tui_gateway.transport import FanoutTransport, current_transport

    prompt = _INGRESS.get()
    _INGRESS.set(None)
    transport = current_transport()
    owner = session.get("transport")
    owns_transport = owner is transport or (
        isinstance(owner, FanoutTransport) and owner.contains(transport))
    if (type(prompt) is not _StdioPrompt or transport is not prompt.transport
            or not owns_transport or session.get("source") != "tui"
            or session.get("auth_user_id") or request_id != prompt.request_id
            or params.get("session_id") != prompt.session_id or params.get("text") != prompt.text
            or not isinstance(text, str) or not text.strip() or not home or not session_id):
        return None
    if read_trusted_tui_session(home, session_id) is not None:
        return None  # Identity is immutable; later turns need no new provenance anchor.
    return TuiSessionReceipt(str(Path(home).resolve()), session_id, text, seal=_SEAL)


def _record_digest(row):
    encoded = json.dumps([CONTRACT, *[row[key] for key in _FIELDS]],
                         separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _session_matches(session, proof):
    return (session is not None and session["source"] == "tui" and not session["user_id"]
            and session["session_key"] == proof["session_key"]
            and session["started_at"] == proof["session_started_at"])


def _plain_user_row(row):
    return (row is not None and row["role"] == "user" and not row["display_kind"]
            and not row["_compressed_summary"] and row["platform_message_id"] is None
            and isinstance(row["message_uid"], str) and bool(row["message_uid"]))


def bind_tui_session_receipt(conn, db_path, session_id, row_id, receipt):
    """Same transaction as append_message; no source enrollment or outbox exists."""
    if (type(receipt) is not TuiSessionReceipt or receipt.seal is not _SEAL
            or receipt.home != str(Path(db_path).resolve().parent)
            or receipt.session_id != session_id):
        return None
    session = conn.execute(
        "SELECT source,user_id,session_key,started_at FROM sessions WHERE id=?", (session_id,)).fetchone()
    row = conn.execute(
        "SELECT role,content,display_kind,_compressed_summary,platform_message_id,message_uid "
        "FROM messages WHERE id=? AND session_id=?", (row_id, session_id)).fetchone()
    if (session is None or session["source"] != "tui" or session["user_id"]
            or not _plain_user_row(row) or row["content"] != receipt.text):
        return None
    proof = dict(receipt_id=receipt.receipt_id, home=receipt.home, session_id=session_id,
                 session_key=session["session_key"], session_started_at=session["started_at"],
                 actor="local-user", message_row_id=row_id, message_uid=row["message_uid"],
                 received_at=receipt.received_at)
    conn.execute("""CREATE TABLE IF NOT EXISTS maintenance_tui_sessions_v1 (
        session_id TEXT PRIMARY KEY, receipt_id TEXT NOT NULL, home TEXT NOT NULL,
        session_key TEXT, session_started_at REAL NOT NULL, actor TEXT NOT NULL,
        message_row_id INTEGER NOT NULL, message_uid TEXT NOT NULL, received_at REAL NOT NULL,
        record_digest TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0)""")
    conn.execute(
        f"INSERT OR IGNORE INTO maintenance_tui_sessions_v1 ({','.join(_FIELDS)},record_digest) "
        f"VALUES ({','.join('?' for _ in range(len(_FIELDS) + 1))})",
        (*[proof[key] for key in _FIELDS], _record_digest(proof)))
    stored = conn.execute(
        "SELECT receipt_id FROM maintenance_tui_sessions_v1 WHERE session_id=? AND revoked=0",
        (session_id,)).fetchone()
    return stored["receipt_id"] if stored is not None else None


def read_trusted_tui_session(profile_home, session_id):
    """Verify the exact durable session/row incarnation; return READ identity only.

    A legacy TUI session gains identity only after a genuine local submit. Queue
    re-placement may retire the anchor; its durable row remains valid. Deleting
    that row, changing its identity/kind or rotating to another session fails
    closed. Content rewrites do not grant authorship or mutation authority.
    A first steer/redirect has no separate durable original row, so it cannot
    establish this identity; a later ordinary or queued local submit can.
    """
    if not isinstance(session_id, str) or not session_id:
        return None
    home = Path(profile_home).resolve()
    try:
        with closing(sqlite3.connect((home / "state.db").as_uri() + "?mode=ro", uri=True, timeout=2)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN")
            proof = db.execute("SELECT * FROM maintenance_tui_sessions_v1 WHERE session_id=? AND revoked=0",
                               (session_id,)).fetchone()
            if (proof is None or proof["home"] != str(home) or proof["actor"] != "local-user"
                    or proof["record_digest"] != _record_digest(proof)):
                return None
            session = db.execute("SELECT source,user_id,session_key,started_at FROM sessions WHERE id=?",
                                 (session_id,)).fetchone()
            row = db.execute(
                "SELECT role,display_kind,_compressed_summary,platform_message_id,message_uid "
                "FROM messages WHERE id=? AND session_id=?", (proof["message_row_id"], session_id)).fetchone()
            if (not _session_matches(session, proof) or not _plain_user_row(row)
                    or row["message_uid"] != proof["message_uid"]):
                return None
            return {"schema": CONTRACT, "platform": "tui", "audience": "private", "capability": "read",
                    **dict(proof)}
    except (sqlite3.Error, OSError, ValueError):
        return None


def revoke_trusted_tui_session(profile_home, session_id, receipt_id):
    """Revoke only the caller's exact receipt; a later submit cannot replace it."""
    home = Path(profile_home).resolve()
    try:
        with closing(sqlite3.connect((home / "state.db").as_uri() + "?mode=rw", uri=True, timeout=5)) as db:
            result = db.execute(
                "UPDATE maintenance_tui_sessions_v1 SET revoked=1 "
                "WHERE session_id=? AND receipt_id=? AND home=? AND revoked=0",
                (session_id, receipt_id, str(home)))
            db.commit()
            return result.rowcount == 1
    except (sqlite3.Error, OSError, ValueError):
        return False
