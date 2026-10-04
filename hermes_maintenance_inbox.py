"""Owner-scoped views of the host's existing SQLite delivery outbox.

No vector client, worker, service startup or remote notification lives here.
Caller IDs must be committed at the CLI or authenticated gateway boundary.
"""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time
import base64
import secrets
from uuid import uuid4

from hermes_maintenance_source import read_committed_source, question_display_hash


def independent_enabled(home):
    try:
        cfg = json.loads((Path(home) / 'maintenance-runtime.json').read_text(encoding='utf-8-sig'))
        return cfg.get('enabled') is True and cfg.get('independent_host') is True
    except (OSError, ValueError, AttributeError):
        return False


def caller_owner(home, source_id):
    home = Path(home).resolve()
    source = read_committed_source(home, source_id)
    cfg = json.loads((home / 'mem0.json').read_text(encoding='utf-8-sig'))
    governed = cfg.get('governed', {})
    platform = {'cli-local': 'cli', 'feishu-human': 'feishu',
                'qqbot-human': 'qqbot', 'weixin-human': 'weixin'}.get((source or {}).get('authority'))
    if (source is None or source['home'] != str(home) or source['origin'] != 'human'
            or source['audience'] != 'private' or governed.get('maintenance_controls') is not True
            or governed.get('maintenance_intake') is not True):
        raise ValueError('MAINTENANCE_INBOX_CALLER_REQUIRED')
    matches = [r['memory_user_id'] for r in governed.get('production', {}).get('owners', [])
               if r.get('platform') == platform and r.get('actor_id') == source['actor']]
    if len(matches) != 1:
        raise ValueError('MAINTENANCE_INBOX_OWNER_REQUIRED')
    return source, matches[0]


def _journal(home, mode='ro'):
    return sqlite3.connect((Path(home).resolve() / 'mem0-governance' / 'production.sqlite3').as_uri()
                           + '?mode=' + mode, uri=True, timeout=5)


def _request_matches(task, source):
    """Mirror the accepted task/source identity checked by MaintenanceControls._target."""
    keys = {'home': 'source_home', 'authority': 'source_authority', 'namespace': 'source_namespace',
            'actor': 'actor', 'revision': 'revision', 'raw_digest': 'raw_digest'}
    return (source is not None and source['source_id'] == (task.get('request_source_id') or task.get('source_id'))
            and all(source.get(key) == task.get(column) for key, column in keys.items()))


def read_inbox(home, source_id, *, task_id=None):
    """Same-owner CLI/gateway contract. Reading alone supplies no display evidence."""
    source, owner = caller_owner(home, source_id)
    try:
        with closing(_journal(home)) as db:
            db.row_factory = sqlite3.Row
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name='maintenance_host_delivery'").fetchone():
                return []
            rows = db.execute('''SELECT d.*,q.delivery_state AS question_state,q.expires_at,
                    t.source_id AS request_source_id,t.revision,t.raw_digest,t.source_home,
                    t.source_authority,t.source_namespace,t.actor,c.cancel_requested_at,
                    w.proposal_receipt
                FROM maintenance_host_delivery d JOIN maintenance_tasks t USING(task_id)
                JOIN maintenance_task_control c USING(task_id)
                LEFT JOIN maintenance_worker_result w USING(task_id)
                LEFT JOIN maintenance_pending_question q USING(ref)
                WHERE d.owner_id=? AND d.state!='CANCELLED'
                ORDER BY d.updated_at,d.ref''', (owner,)).fetchall()
        visible = []
        for raw in rows:
            row = dict(raw)
            if task_id is not None and row['task_id'] != task_id:
                continue
            request = read_committed_source(home, row['request_source_id'])
            if not _request_matches(row, request) or row['cancel_requested_at'] is not None:
                continue
            try:
                _, current_owner = caller_owner(home, row['request_source_id'])
            except ValueError:
                continue
            if current_owner != owner or request['audience'] != 'private':
                continue
            payload = json.loads(row['payload_json'])
            if row['kind'] == 'QUESTION' and (row['question_state'] not in ('PREPARED', 'DELIVERED')
                    or row['expires_at'] <= time.time()
                    or row['proposal_receipt'] != 'reasoning:' + row['task_id'] + ':' + payload['proposal_revision']):
                continue
            visible.append({'ref': row['ref'], 'task_id': row['task_id'], 'kind': row['kind'],
                            'payload': payload, 'channel': row['channel'], 'session_key': row['session_key'],
                            'request_source_id': row['request_source_id'], 'evidence_level': row['evidence_level'],
                            'receipt': json.loads(row['receipt_json']) if row['receipt_json'] else None})
        return visible
    except sqlite3.OperationalError as exc:
        if 'unable to open database' in str(exc):
            return []
        raise


def display_inbox(home, source_id, display, *, ref=None):
    """Call a trusted CLI display function, then commit local DISPLAYED evidence.

    Success means the display call returned normally; it makes no reading claim.
    A crash after printing but before commit leaves the row available next time.
    """
    source, owner = caller_owner(home, source_id)
    if source['authority'] != 'cli-local' or source.get('control_kind') != 'maintenance_inbox_read':
        raise ValueError('MAINTENANCE_INBOX_LOCAL_CALLER_REQUIRED')
    shown = []
    for row in read_inbox(home, source_id):
        if row['channel'] != 'local-inbox':
            continue
        if ref is not None and row['ref'] != ref:
            continue
        if ref is None and row['kind'] == 'RESULT' and row['evidence_level'] == 'DISPLAYED':
            continue
        payload = row['payload']
        body = row['ref'] + '\n' + payload['question']
        if row['kind'] == 'RESULT':
            body += '\n' + json.dumps(payload.get('result'), ensure_ascii=False, sort_keys=True)
        if display(body) is False:
            raise ValueError('MAINTENANCE_INBOX_DISPLAY_FAILED')
        # Recheck source, revocation, cancellation and the frozen revision after rendering.
        current = next((r for r in read_inbox(home, source_id) if r['ref'] == row['ref']), None)
        if current is None or current['payload'] != payload:
            raise ValueError('MAINTENANCE_INBOX_STALE')
        receipt = {'channel': 'local-inbox', 'session_key': 'owner:' + owner,
                   'message_ref': None, 'evidence_level': 'DISPLAYED',
                   'display_ref': 'CLI-' + uuid4().hex, 'display_source_id': source_id,
                   'display_session_id': source['session_id'],
                   'display_hash': question_display_hash(payload['question']), 'delivered_at': time.time()}
        with closing(_journal(home, 'rw')) as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute("""UPDATE maintenance_host_delivery SET state='ACKNOWLEDGED',
                evidence_level='DISPLAYED',receipt_json=?,updated_at=?
                WHERE ref=? AND owner_id=? AND channel='local-inbox'
                AND state IN ('PREPARED','UNCERTAIN')""",
                (json.dumps(receipt, sort_keys=True, separators=(',', ':')), time.time(), row['ref'], owner))
            db.commit()
        shown.append(row['ref'])
    return shown


def verify_display_receipt(home, owner, payload, receipt):
    try:
        source, current_owner = caller_owner(home, receipt['display_source_id'])
        return (current_owner == owner and source['authority'] == 'cli-local'
                and source.get('control_kind') == 'maintenance_inbox_read'
                and receipt['display_session_id'] == source['session_id']
                and receipt['session_key'] == 'owner:' + owner
                and receipt['channel'] == 'local-inbox' and receipt.get('message_ref') is None
                and receipt.get('evidence_level') == 'DISPLAYED'
                and receipt['display_hash'] == question_display_hash(payload['question'])
                and isinstance(receipt.get('display_ref'), str) and receipt['display_ref'].startswith('CLI-'))
    except (KeyError, ValueError, OSError):
        return False


def answer_binding(home, source_id, *, ref=None, consent=False, text=None):
    """Bare answers select only a unique prompt displayed in this CLI session."""
    source, _ = caller_owner(home, source_id)
    if source['authority'] != 'cli-local':
        if ref is None or text is None:
            raise ValueError('MAINTENANCE_INBOX_EXACT_DISPLAYED_QUESTION_REQUIRED')
        native = next((r for r in read_inbox(home, source_id) if r['ref'] == ref and r['kind'] == 'QUESTION'
                       and r['channel'] == source['authority'].removesuffix('-human')), None)
        if native is not None:
            request, _ = caller_owner(home, native['request_source_id'])
            if not _same_entry(request, source):
                raise ValueError('MAINTENANCE_INBOX_EXACT_DISPLAYED_QUESTION_REQUIRED')
            if source['authority'] == 'weixin-human':
                from hermes_maintenance_source import parse_weixin_question_reply
                if parse_weixin_question_reply(text, native['payload'].get('reply_code'), native['payload']['kind']) is None:
                    raise ValueError('USER_ACK_BINDING_REQUIRED')
            elif not native['receipt'] or not native['receipt'].get('message_ref'):
                raise ValueError('MAINTENANCE_INBOX_EXACT_DISPLAYED_QUESTION_REQUIRED')
            return {k: native['payload'][k] for k in ('task_id', 'item_id', 'proposal_revision')}
        projected = answer_projection(home, source_id, ref=ref, text=ref + ' ' + text)
        if projected is None:
            raise ValueError('MAINTENANCE_INBOX_EXACT_DISPLAYED_QUESTION_REQUIRED')
        return {k: projected['payload'][k] for k in ('task_id', 'item_id', 'proposal_revision')}
    rows = [r for r in read_inbox(home, source_id) if r['kind'] == 'QUESTION'
            and (not consent or r['payload']['kind'] == 'CONFIRM')
            and r['evidence_level'] == 'DISPLAYED' and r['receipt']
            and verify_display_receipt(home, caller_owner(home, source_id)[1], r['payload'], r['receipt'])
            and (r['ref'] == ref if ref is not None else
                 r['receipt'].get('display_session_id') == source['session_id'])]
    if len(rows) != 1:
        raise ValueError('MAINTENANCE_INBOX_EXACT_DISPLAYED_QUESTION_REQUIRED')
    payload = rows[0]['payload']
    return {k: payload[k] for k in ('task_id', 'item_id', 'proposal_revision')}


def read_status(home, source_id, *, task_id=None):
    """Business progress is independent of whether a result notification exists."""
    _, owner = caller_owner(home, source_id)
    try:
        with closing(_journal(home)) as db:
            db.row_factory = sqlite3.Row
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name='maintenance_tasks'").fetchone():
                return []
            task_filter = ' AND t.task_id=?' if task_id is not None else ''
            rows = db.execute('''SELECT t.*,c.stage,c.cancel_requested_at,
                w.status AS worker_status,w.result_json FROM maintenance_tasks t
                JOIN maintenance_task_control c USING(task_id)
                LEFT JOIN maintenance_worker_result w USING(task_id)
                WHERE t.owner_id=?''' + task_filter + ' ORDER BY t.created_at DESC,t.task_id DESC LIMIT 50',
                (owner, task_id) if task_id is not None else (owner,)).fetchall()
            result = []
            for row in rows:
                if task_id is not None and row['task_id'] != task_id:
                    continue
                try:
                    request, mapped = caller_owner(home, row['source_id'])
                except ValueError:
                    continue
                if mapped != owner or not _request_matches(dict(row), request):
                    continue
                business = row['worker_status'] or row['state']
                progress = db.execute("SELECT 1 FROM sqlite_master WHERE name='maintenance_host_progress'").fetchone()
                if progress:
                    p = db.execute('SELECT status FROM maintenance_host_progress WHERE task_id=?', (row['task_id'],)).fetchone()
                    if p:
                        business = p[0]
                if row['state'] == 'AUTHORITY_INVALID':
                    business = row['state']
                label = {'COMPLETE': '已提交', 'PARTIAL': '部分完成', 'PARTIAL_COMPLETE': '部分完成',
                         'WAIT_CLARIFICATION': '等待回答', 'WAIT_CONFIRMATION': '等待回答',
                         'WAITING': '等待回答'}.get(business, business)
                notifications = []
                if db.execute("SELECT 1 FROM sqlite_master WHERE name='maintenance_host_delivery'").fetchone():
                    notices = db.execute('SELECT ref,channel,evidence_level FROM maintenance_host_delivery WHERE task_id=? AND owner_id=?',
                                         (row['task_id'], owner)).fetchall()
                    notifications = [dict(n) for n in notices]
                    if db.execute("SELECT 1 FROM sqlite_master WHERE name='maintenance_inbox_projection'").fetchone():
                        for n in notices:
                            for p in db.execute('SELECT source_id,evidence_level FROM maintenance_inbox_projection WHERE ref=?', (n['ref'],)):
                                try:
                                    viewer, viewer_owner = caller_owner(home, p['source_id'])
                                except ValueError:
                                    continue
                                if viewer_owner == owner:
                                    notifications.append({'ref': n['ref'], 'channel': viewer['authority'].removesuffix('-human'),
                                                          'evidence_level': p['evidence_level']})
                result.append({'task_id': row['task_id'], 'status': business, 'label': label,
                               'stage': row['stage'], 'cancel_requested': row['cancel_requested_at'] is not None,
                               'notifications': notifications,
                               'result': json.loads(row['result_json']) if row['result_json'] else None})
        return result
    except sqlite3.OperationalError as exc:
        if 'unable to open database' in str(exc):
            return []
        raise


def format_status(rows):
    return '\n'.join(r['task_id'] + ' 业务：' + r['label'] + ('（取消请求已记录）' if r['cancel_requested'] else '')
                     + '\n通知证据：' + (', '.join(n['channel'] + '=' + str(n['evidence_level'] or 'PREPARED')
                                                   for n in r['notifications']) or '无通知回执')
                     for r in rows) or '没有可用维护任务。'


def _projection_schema(db):
    db.execute('''CREATE TABLE IF NOT EXISTS maintenance_inbox_projection (
        ref TEXT NOT NULL,source_id TEXT NOT NULL,payload_json TEXT NOT NULL,
        body TEXT NOT NULL,display_hash TEXT NOT NULL,reply_code TEXT,
        evidence_level TEXT NOT NULL,receipt_json TEXT,
        PRIMARY KEY(ref,source_id))''')


def _same_entry(a, b):
    return all(a.get(k) == b.get(k) for k in
               ('authority', 'actor', 'namespace', 'session_id', 'session_key'))


def _projection_body(ref, payload, code=None):
    question = payload['question'].removesuffix('回复“同意”或“拒绝”。') if code else payload['question']
    body = ref + '\n' + question
    if code:
        instruction = f'确认 {code}' if payload['kind'] == 'CONFIRM' else f'回答 {code}：你的回答'
        body += f"\n编号：{code}\n到期：{payload['expires_at']}\n请精确回复：{instruction}"
    return body


def prepare_projection(home, source_id, row):
    """Freeze a view of the existing outbox before a single gateway send."""
    source, _ = caller_owner(home, source_id)
    current = next((r for r in read_inbox(home, source_id) if r['ref'] == row['ref']), None)
    # The resident host may advance receipt evidence between these reads.
    # Revalidate the frozen content and identity, independent of that evidence.
    frozen = ('ref', 'task_id', 'kind', 'payload', 'channel', 'session_key', 'request_source_id')
    if (source['authority'] not in ('feishu-human', 'qqbot-human', 'weixin-human')
            or source.get('control_kind') != 'maintenance_inbox_read'
            or current is None or any(row.get(k) != current.get(k) for k in frozen)):
        raise ValueError('MAINTENANCE_PROJECTION_CALLER_REQUIRED')
    payload = row['payload']
    code = None
    body = _projection_body(row['ref'], payload)
    if row['kind'] == 'RESULT':
        body += '\n' + json.dumps(payload.get('result'), ensure_ascii=False, sort_keys=True)
    elif source['authority'] == 'weixin-human' and row['channel'] == 'local-inbox':
        code = 'WX-' + base64.b32encode(secrets.token_bytes(10)).decode('ascii')
        body = _projection_body(row['ref'], payload, code)
    with closing(_journal(home, 'rw')) as db:
        _projection_schema(db)
        db.execute('INSERT OR IGNORE INTO maintenance_inbox_projection VALUES (?,?,?,?,?,?,?,NULL)',
                   (row['ref'], source_id, json.dumps(payload, sort_keys=True), body,
                    question_display_hash(body), code, 'PREPARED'))
        db.row_factory = sqlite3.Row
        projection = dict(db.execute('SELECT * FROM maintenance_inbox_projection WHERE ref=? AND source_id=?',
                                     (row['ref'], source_id)).fetchone())
        db.commit()
    return projection


def begin_projection_send(home, source_id, ref):
    caller_owner(home, source_id)
    with closing(_journal(home, 'rw')) as db:
        changed = db.execute("UPDATE maintenance_inbox_projection SET evidence_level='DELIVERY_UNKNOWN' WHERE ref=? AND source_id=? AND evidence_level='PREPARED'",
                             (ref, source_id)).rowcount
        db.commit()
    return changed == 1


def record_projection(home, source_id, ref, sent):
    """Gateway API receipts are remote IDs or ACKs, never local DISPLAYED."""
    source, _ = caller_owner(home, source_id)
    if not sent.success:
        return False
    if source['authority'] == 'weixin-human':
        level, message = 'API_ACCEPTED', None
    else:
        level, message = 'REMOTE_MESSAGE_ID', sent.message_id
        if not isinstance(message, str) or not message:
            return False
    receipt = {'channel': source['authority'].removesuffix('-human'),
               'session_key': source['session_key'], 'message_ref': message,
               'evidence_level': level, 'delivered_at': time.time(), 'projection_source_id': source_id,
               'projection_ref': ref}
    with closing(_journal(home, 'rw')) as db:
        db.execute('UPDATE maintenance_inbox_projection SET evidence_level=?,receipt_json=? WHERE ref=? AND source_id=?',
                   (level, json.dumps(receipt, sort_keys=True), ref, source_id))
        db.commit()
    return True


def answer_projection(home, source_id, *, ref=None, text=None):
    """Revalidate exact content, current owner/revision/expiry and the receiving session.

    An ACK-only Weixin send requires the explicit coded human reply. A remote ID
    permits the exact ref in that same receiving conversation, never bare consent.
    """
    committed = read_committed_source(home, source_id)
    raw = text if text is not None else (committed or {}).get('raw_text', '')
    if ref is None:
        ref = raw.split(' ', 1)[0]
    if not ref.startswith('Q-') or not raw.startswith(ref + ' '):
        return None
    try:
        source, owner = caller_owner(home, source_id)
    except (ValueError, OSError):
        return None
    visible = next((r for r in read_inbox(home, source_id) if r['ref'] == ref and r['kind'] == 'QUESTION'), None)
    if visible is None or visible['channel'] != 'local-inbox':
        return None
    return _projection_answer(home, source, owner, ref, raw, visible['payload'], time.time())


def _projection_answer(home, source, owner, ref, raw, expected_payload, when):
    from hermes_maintenance_source import parse_weixin_question_reply
    with closing(_journal(home)) as db:
        db.row_factory = sqlite3.Row
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='maintenance_inbox_projection'").fetchone():
            return None
        rows = db.execute('SELECT * FROM maintenance_inbox_projection WHERE ref=?', (ref,)).fetchall()
    for r in rows:
        try:
            viewer, viewer_owner = caller_owner(home, r['source_id'])
        except ValueError:
            continue
        payload = json.loads(r['payload_json'])
        if (viewer_owner != owner or not _same_entry(viewer, source)
                or viewer.get('control_kind') != 'maintenance_inbox_read'
                or payload != expected_payload or payload['expires_at'] <= when
                or r['display_hash'] != question_display_hash(r['body'])):
            continue
        binding = source.get('control_binding_json')
        if source.get('control_kind') == 'typed_clarify_response' and (
                not binding or json.loads(binding) != {k: payload[k] for k in ('task_id', 'item_id', 'proposal_revision')}):
            continue
        body = _projection_body(ref, payload, r['reply_code'] if source['authority'] == 'weixin-human' else None)
        if body != r['body']:
            continue
        response = raw[len(ref):].strip()
        receipt = json.loads(r['receipt_json']) if r['receipt_json'] else None
        if source['authority'] == 'weixin-human':
            if r['evidence_level'] not in ('DELIVERY_UNKNOWN', 'API_ACCEPTED', 'USER_ACKNOWLEDGED'):
                continue
            response = parse_weixin_question_reply(response, r['reply_code'], payload['kind'])
            if response is None:
                continue
            # The exact reply authenticates observation even if the send ACK was lost.
            receipt = {'channel': 'weixin', 'session_key': source['session_key'], 'message_ref': None,
                       'evidence_level': 'USER_ACKNOWLEDGED', 'response_source_id': source['source_id'],
                       'delivered_at': source['received_time'], 'projection_source_id': r['source_id'], 'projection_ref': ref}
        elif (r['evidence_level'] != 'REMOTE_MESSAGE_ID' or receipt is None or not receipt.get('message_ref')
              or receipt.get('projection_source_id') != r['source_id'] or receipt.get('projection_ref') != ref
              or receipt.get('channel') != source['authority'].removesuffix('-human')
              or receipt.get('session_key') != source['session_key']):
            continue
        receipt = dict(receipt, display_hash=r['display_hash'])
        return {'payload': payload, 'receipt': receipt, 'response': response}
    return None


def acknowledge_projection(home, source_id):
    """The host records a verified coded human observation, separately from API ACK."""
    projected = answer_projection(home, source_id)
    if projected is None or projected['receipt']['evidence_level'] != 'USER_ACKNOWLEDGED':
        return
    receipt = projected['receipt']
    with closing(_journal(home, 'rw')) as db:
        db.execute("UPDATE maintenance_inbox_projection SET evidence_level='USER_ACKNOWLEDGED',receipt_json=? WHERE ref=? AND source_id=? AND evidence_level IN ('DELIVERY_UNKNOWN','API_ACCEPTED')",
                   (json.dumps(receipt, sort_keys=True), receipt['projection_ref'], receipt['projection_source_id']))
        db.commit()


def resolved_projection_response(home, source, row):
    """Replay the frozen observation at its recorded resolution time."""
    _, owner = caller_owner(home, source['source_id'])
    payload = {k: row[k] for k in ('ref', 'owner_id', 'task_id', 'item_id', 'proposal_revision', 'kind', 'expires_at')}
    payload.update(question=row['question_text'], choices=json.loads(row['choices_json']) if row.get('choices_json') else None)
    if row.get('reply_code'):
        payload.update(reply_code=row['reply_code'], display_hash=row['display_hash'])
    projected = _projection_answer(home, source, owner, row['ref'], source['raw_text'], payload, row['resolved_at'])
    return projected['response'] if projected else None


def projected_reply_ref(home, source_id, text):
    """Only exact coded Weixin replies can select an off-session projection."""
    source, owner = caller_owner(home, source_id)
    if source['authority'] != 'weixin-human':
        return None
    try:
        with closing(_journal(home)) as db:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name='maintenance_inbox_projection'").fetchone():
                return None
            refs = [r[0] for r in db.execute('SELECT DISTINCT ref FROM maintenance_inbox_projection WHERE reply_code IS NOT NULL')]
    except sqlite3.OperationalError as exc:
        if 'unable to open database' in str(exc):
            return None
        raise
    candidates = [ref for ref in refs if answer_projection(home, source_id, ref=ref, text=ref + ' ' + text)]
    return candidates[0] if len(candidates) == 1 else None
