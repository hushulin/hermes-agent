"""Ordinary assent is chat unless this CLI actually displayed a live question."""
import json
import os
import sqlite3
import time
from types import SimpleNamespace

import pytest

from hermes_cli import maintenance_host as host
from hermes_cli.maintenance_input import commit_control
from hermes_maintenance_inbox import display_inbox
from hermes_maintenance_source import committed_source_for_receipt, read_committed_source
from hermes_state import SessionDB


@pytest.fixture
def world(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    (home / 'mem0.json').write_text(json.dumps({'governed': {
        'maintenance_intake': True, 'maintenance_controls': True,
        'production': {'owners': [{'platform': 'cli', 'actor_id': 'local-user',
                                  'memory_user_id': 'owner-1'}]}}}), encoding='utf-8')
    (home / 'maintenance-runtime.json').write_text(json.dumps({
        'enabled': True, 'mode': 'isolated-test', 'independent_host': True}), encoding='utf-8')
    db = SessionDB(db_path=home / 'state.db')
    messages = []
    cli = SimpleNamespace(session_id='chat', _session_db=db, _console_print=messages.append)
    result = SimpleNamespace(home=home, db=db, cli=cli, messages=messages,
                             journal=home / 'mem0-governance' / 'production.sqlite3')
    try:
        yield result
    finally:
        db.close()


def answer_controls(world):
    with sqlite3.connect(world.home / 'state.db') as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='maintenance_local_controls_v1'").fetchone():
            return []
        source_ids = [row[0] for row in db.execute(
            "SELECT source_id FROM maintenance_local_controls_v1 WHERE control_kind='typed_clarify_response'")]
    return [read_committed_source(world.home, source_id) for source_id in source_ids]


def question(world, ref='Q-one', *, shown=True, owner='owner-1', kind='CONFIRM'):
    receipt = commit_control(world.cli, 'slash_queue', 'Review database migration')
    source = committed_source_for_receipt(world.home, world.cli.session_id, receipt)
    world.journal.parent.mkdir(exist_ok=True)
    payload = {'kind': kind, 'task_id': 'task-' + ref, 'item_id': 'item-' + ref,
               'proposal_revision': 'proposal-1', 'question': 'Change MySQL to PostgreSQL?'}
    with sqlite3.connect(world.journal) as db:
        db.executescript('''
            CREATE TABLE IF NOT EXISTS maintenance_tasks (
                task_id TEXT PRIMARY KEY,source_id TEXT,revision TEXT,raw_digest TEXT,
                source_home TEXT,source_authority TEXT,source_namespace TEXT,actor TEXT);
            CREATE TABLE IF NOT EXISTS maintenance_task_control (task_id TEXT PRIMARY KEY,cancel_requested_at INTEGER);
            CREATE TABLE IF NOT EXISTS maintenance_worker_result (task_id TEXT PRIMARY KEY,proposal_receipt TEXT);
            CREATE TABLE IF NOT EXISTS maintenance_pending_question (ref TEXT PRIMARY KEY,delivery_state TEXT,expires_at REAL);
            CREATE TABLE IF NOT EXISTS maintenance_host_delivery (
                ref TEXT PRIMARY KEY,owner_id TEXT,task_id TEXT,kind TEXT,state TEXT,
                payload_json TEXT,receipt_json TEXT,channel TEXT,session_key TEXT,evidence_level TEXT,updated_at REAL);
        ''')
        db.execute('INSERT INTO maintenance_tasks VALUES (?,?,?,?,?,?,?,?)', (
            payload['task_id'], source['source_id'], source['revision'], source['raw_digest'],
            source['home'], source['authority'], source['namespace'], source['actor']))
        db.execute('INSERT INTO maintenance_task_control VALUES (?,NULL)', (payload['task_id'],))
        db.execute('INSERT INTO maintenance_worker_result VALUES (?,?)', (
            payload['task_id'], 'reasoning:' + payload['task_id'] + ':proposal-1'))
        db.execute('INSERT INTO maintenance_pending_question VALUES (?,?,?)', (ref, 'PREPARED', time.time() + 3600))
        db.execute('INSERT INTO maintenance_host_delivery VALUES (?,?,?,?,?,?,?,?,?,?,?)', (
            ref, owner, payload['task_id'], 'QUESTION', 'PREPARED', json.dumps(payload),
            None, 'local-inbox', 'owner:' + owner, 'INBOX_STORED', time.time()))
    if shown:
        home, caller = host._caller(world.cli)
        display_inbox(home, caller, lambda text: None, ref=ref)
    return payload


@pytest.mark.parametrize('raw', ['确认', '同意', '拒绝', '不同意', 'yes', 'agree', 'no', 'reject', ' YES ', '\n确认\t'])
def test_empty_inbox_falls_through_without_answer(world, raw):
    assert host.consume_bare_consent(world.cli, raw) is False
    assert world.messages == []
    assert answer_controls(world) == []
    assert not world.journal.exists()


@pytest.mark.parametrize('raw', ['确认。', '请确认', '取消', 'sure', 'ok', 'はい', '/maintenance inbox', '/maintenance cancel task'])
def test_other_text_is_not_maintenance_consent(world, raw):
    question(world)
    assert host.consume_bare_consent(world.cli, raw) is False
    assert world.messages == []
    assert answer_controls(world) == []


@pytest.mark.parametrize('raw', ['确认', '拒绝', ' YES ', 'no'])
def test_unique_displayed_question_keeps_exact_binding(world, raw):
    expected = question(world)
    assert host.consume_bare_consent(world.cli, raw) is True
    controls = answer_controls(world)
    assert len(controls) == 1
    assert json.loads(controls[0]['control_binding_json']) == {
        key: expected[key] for key in ('task_id', 'item_id', 'proposal_revision')}
    assert controls[0]['raw_text'] == raw.strip()
    assert world.messages == ['回答已提交，维护运行时将重验并处理。']


@pytest.mark.parametrize('state', ['unshown', 'other-session', 'other-owner', 'expired', 'cancelled',
                                  'resolved', 'stale-revision', 'invalid-display', 'clarification'])
def test_no_eligible_question_is_chat_but_explicit_answer_stays_strict(world, state):
    question(world, shown=state != 'unshown', owner='other-owner' if state == 'other-owner' else 'owner-1',
             kind='CLARIFY' if state == 'clarification' else 'CONFIRM')
    if state == 'other-session':
        world.cli.session_id = 'another-chat'
    statements = {
        'expired': "UPDATE maintenance_pending_question SET expires_at=0",
        'cancelled': "UPDATE maintenance_task_control SET cancel_requested_at=1",
        'resolved': "UPDATE maintenance_pending_question SET delivery_state='ANSWERED'",
        'stale-revision': "UPDATE maintenance_worker_result SET proposal_receipt='reasoning:task-Q-one:proposal-2'",
        'invalid-display': "UPDATE maintenance_host_delivery SET receipt_json=json_set(receipt_json,'$.display_hash','invalid')",
    }
    if state in statements:
        with sqlite3.connect(world.journal) as db:
            db.execute(statements[state])
    assert host.consume_bare_consent(world.cli, '确认') is False
    assert world.messages == []
    assert answer_controls(world) == []
    # Explicit refs retain their established cross-CLI-session semantics: an
    # exact Q reference can address a verified display from another session.
    command = '/maintenance answer 确认' if state == 'other-session' else '/maintenance answer Q-one 确认'
    with pytest.raises(ValueError, match='EXACT_DISPLAYED_QUESTION_REQUIRED'):
        host.handle_command(world.cli, command)
    assert answer_controls(world) == []


def test_two_displayed_questions_fail_closed(world):
    question(world)
    question(world, 'Q-two')
    assert host.consume_bare_consent(world.cli, '确认') is True
    assert world.messages == ['MAINTENANCE_INBOX_EXACT_DISPLAYED_QUESTION_REQUIRED']
    assert answer_controls(world) == []


def test_unreadable_ledger_never_becomes_chat_or_authority(world):
    question(world)
    with sqlite3.connect(world.journal) as db:
        db.execute('DROP TABLE maintenance_pending_question')
    with pytest.raises(sqlite3.OperationalError):
        host.consume_bare_consent(world.cli, '确认')
    assert answer_controls(world) == []


@pytest.mark.platforms('posix')
def test_existing_permission_denied_journal_never_becomes_chat(world):
    if os.geteuid() == 0:
        pytest.skip('Root can read files despite permission bits.')
    question(world)
    mode = world.journal.stat().st_mode
    world.journal.chmod(0)
    try:
        with pytest.raises(sqlite3.OperationalError):
            host.consume_bare_consent(world.cli, '确认')
        assert answer_controls(world) == []
        with pytest.raises(ValueError, match='EXACT_DISPLAYED_QUESTION_REQUIRED'):
            host.handle_command(world.cli, '/maintenance answer Q-one 确认')
    finally:
        world.journal.chmod(mode)


def test_selected_question_is_rechecked_without_rebinding(world, monkeypatch):
    question(world)
    original = host.answer_binding
    calls = []

    def changed_after_selection(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        if len(calls) == 1:
            with sqlite3.connect(world.journal) as db:
                db.execute("UPDATE maintenance_task_control SET cancel_requested_at=1")
            question(world, 'Q-replacement')
        return result

    monkeypatch.setattr(host, 'answer_binding', changed_after_selection)
    assert host.consume_bare_consent(world.cli, '确认') is True
    assert len(calls) == 2
    assert calls[0] != calls[1]
    assert world.messages == ['MAINTENANCE_INBOX_EXACT_DISPLAYED_QUESTION_REQUIRED']
    assert answer_controls(world) == []


def test_explicit_missing_reference_always_fails_closed(world):
    with pytest.raises(ValueError, match='EXACT_DISPLAYED_QUESTION_REQUIRED'):
        host.handle_command(world.cli, '/maintenance answer Q-missing 确认')
    assert answer_controls(world) == []


@pytest.mark.parametrize('raw', ['确认', ' YES ', '拒绝'])
def test_real_tui_dispatch_reaches_chat_without_active_question(world, raw):
    from hermes_cli.cli_tui_runtime_mixin import CLITuiRuntimeMixin
    from hermes_maintenance_source import current_admitted_source

    chat_calls = []

    def chat(message, **kwargs):
        source = current_admitted_source()
        chat_calls.append((message, source.raw, source.control_binding_json))

    cli = world.cli
    cli._tui_unwrap_input = lambda value: (value, False, False)
    cli._typed_voice_stop = lambda value: False
    cli.handle_bang_shell = lambda value: False
    cli._pending_resume_sessions = None
    cli._print_user_message_preview = lambda value: None
    cli._turn_summary_begin = lambda: None
    cli._app = SimpleNamespace(invalidate=lambda: None)
    cli._tui_after_turn = lambda: None
    cli.chat = chat
    CLITuiRuntimeMixin._tui_process_one_input(cli, raw)
    assert chat_calls == [(raw, raw, None)]
    assert answer_controls(world) == []


def test_explicit_reference_cannot_use_optional_missing_mode(world):
    home, caller = host._caller(world.cli)
    with pytest.raises(ValueError, match='EXACT_DISPLAYED_QUESTION_REQUIRED'):
        host.answer_binding(home, caller, ref='Q-missing', consent=True, allow_missing=True)


def test_explicit_answer_uses_exact_ref_among_multiple_questions(world):
    question(world)
    expected = question(world, 'Q-two')
    rendered = []
    host.handle_command(world.cli, '/maintenance answer Q-two 确认', display=rendered.append)
    controls = answer_controls(world)
    assert len(controls) == 1
    assert json.loads(controls[0]['control_binding_json'])['task_id'] == expected['task_id']
    assert rendered == ['回答已提交，维护运行时将重验并处理。']
    assert world.messages == []


def test_disabled_runtime_leaves_consent_as_chat(world):
    (world.home / 'maintenance-runtime.json').write_text('{"enabled": false}', encoding='utf-8')
    assert host.consume_bare_consent(world.cli, '确认') is False
    assert world.messages == []
    assert answer_controls(world) == []
