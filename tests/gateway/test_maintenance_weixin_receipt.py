"""Frozen host selection and strict intent grammar; no network or model calls."""
import sqlite3
import time

import pytest

from hermes_maintenance_source import (
    begin_host_question_delivery, host_question_delivery_for_session,
    parse_weixin_question_reply, question_display_hash,
)


CODE = 'WX-ABCDEFGHIJKLMNOP'


@pytest.mark.parametrize('text', [
    '同意', '不要确认 ' + CODE, '他说确认 ' + CODE,
    '“确认 ' + CODE + '”', '确认 ' + CODE + ' extra',
    '确认 ' + CODE.lower(), '确认 WX-AAAAAAAAAAAAAAAA',
    '确认 ' + CODE + '\n拒绝', '确认 ' + CODE[:-1],
])
def test_code_is_never_a_substring_authorization(text):
    assert parse_weixin_question_reply(text, CODE, 'CONFIRM') is None


def test_exact_intent_and_question_kind():
    assert parse_weixin_question_reply('确认 ' + CODE, CODE, 'CONFIRM') == '确认'
    assert parse_weixin_question_reply('拒绝 ' + CODE, CODE, 'CONFIRM') == '拒绝'
    assert parse_weixin_question_reply('不同意 ' + CODE, CODE, 'CONFIRM') == '不同意'
    assert parse_weixin_question_reply('回答 ' + CODE + '：project', CODE, 'CLARIFY') == 'project'
    assert parse_weixin_question_reply('确认 ' + CODE, CODE, 'CLARIFY') is None
    assert parse_weixin_question_reply('回答 ' + CODE + '：project', CODE, 'CONFIRM') is None


def test_unknown_multiple_candidates_expiry_and_owner_session_isolation(tmp_path):
    sqlite3.connect(tmp_path / 'state.db').close()
    expiry = time.time() + 100
    for i, code in enumerate((CODE, 'WX-QRSTUVWXYZ234567', 'WX-AAAAAAAAAAAAAAAA')):
        body = 'private body ' + str(i) + '\n' + code
        begin_host_question_delivery(tmp_path, 'q' + str(i), 'owner' + str(i),
            'alice-session' if i < 2 else 'mallory-session', body, None,
            {'task_id': 'task' + str(i), 'item_id': 'item', 'proposal_revision': 'rev' + str(i)},
            expires_at=expiry, reply_code=code, display_hash=question_display_hash(body),
            question_kind='CONFIRM')
    row, count = host_question_delivery_for_session(tmp_path, 'alice-session')
    assert row is None and count == 2
    row, count = host_question_delivery_for_session(tmp_path, 'alice-session', reply_text='确认 ' + CODE)
    assert row['prompt_id'] == 'q0' and count == 1 and row['state'] == 'PREPARED'
    assert row['message_ref'] is None
    assert host_question_delivery_for_session(tmp_path, 'mallory-session', reply_text='确认 ' + CODE) == (None, 0)
    assert host_question_delivery_for_session(tmp_path, 'alice-session', reply_text='同意') == (None, 0)
    assert host_question_delivery_for_session(tmp_path, 'alice-session', now=expiry + 1,
        reply_text='确认 ' + CODE) == (None, 0)
    with sqlite3.connect(tmp_path / 'state.db') as db:
        db.execute("UPDATE maintenance_host_questions_v1 SET question='tampered' WHERE prompt_id='q0'")
    assert host_question_delivery_for_session(tmp_path, 'alice-session', reply_text='确认 ' + CODE) == (None, 0)


def test_additive_schema_accepts_legacy_rows(tmp_path):
    with sqlite3.connect(tmp_path / 'state.db') as db:
        db.execute('''CREATE TABLE maintenance_host_questions_v1 (
            prompt_id TEXT PRIMARY KEY, owner_id TEXT, session_key TEXT, question TEXT,
            choices_json TEXT, control_binding_json TEXT, state TEXT, created_at REAL,
            expires_at REAL, channel TEXT, message_ref TEXT, delivered_at REAL, delivery_receipt_json TEXT)''')
    begin_host_question_delivery(tmp_path, 'legacy', 'alice', 'session', 'question', None,
        {'task_id': 'task', 'item_id': 'item', 'proposal_revision': 'revision'}, expires_at=time.time() + 100)
    assert host_question_delivery_for_session(tmp_path, 'session') == (None, 0)
