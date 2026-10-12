"""CLI caller boundary for the owner runtime maintenance inbox."""
import argparse
import json
import shlex
from types import SimpleNamespace

from hermes_cli.maintenance_input import commit_control
from hermes_constants import get_hermes_home
from hermes_maintenance_inbox import (answer_binding, display_inbox,
                                      inbox_enabled, read_inbox, read_status, format_status)
from hermes_maintenance_source import (acknowledge_control_source,
                                       committed_source_for_receipt)


def _caller(cli):
    home = get_hermes_home()
    if not inbox_enabled(home):
        raise ValueError('MAINTENANCE_INDEPENDENT_HOST_DISABLED')
    receipt = commit_control(cli, 'maintenance_inbox_read', '/maintenance inbox')
    source = committed_source_for_receipt(home, cli.session_id, receipt)
    if source is None:
        raise ValueError('MAINTENANCE_INBOX_CALLER_REQUIRED')
    acknowledge_control_source(home, source['source_id'], 'inbox-read:' + source['source_id'])
    return home, source['source_id']


def handle_command(cli, command, *, display=None):
    """Used by HermesCLI.process_command and the lightweight local terminal entry."""
    parts = shlex.split(command)
    args = parts[1:]
    operation = args.pop(0) if args else 'inbox'
    render = display or cli._console_print
    home, caller = _caller(cli)
    if operation == 'status':
        rows = read_status(home, caller, task_id=args[0] if args else None)
        render(format_status(rows))
        return rows
    if operation == 'cancel':
        if len(args) != 1 or not any(r['task_id'] == args[0] for r in read_status(home, caller, task_id=args[0])):
            raise ValueError('OWNER_TASK_UNAVAILABLE')
        receipt = commit_control(cli, 'maintenance_cancel', args[0])
        source = committed_source_for_receipt(home, cli.session_id, receipt)
        if source is None:
            raise ValueError('MAINTENANCE_CANCEL_COMMIT_FAILED')
        render('取消请求已提交，维护运行时将重验；这不表示运行中的操作已停止。')
        return source['source_id']
    if operation in ('inbox', 'result'):
        if operation == 'result':
            task = args[0] if args else None
            render(format_status(read_status(home, caller, task_id=task)))
            rows = [r for r in read_inbox(home, caller, task_id=task) if r['kind'] == 'RESULT']
            for row in rows:
                display_inbox(home, caller, render, ref=row['ref'])
            return rows
        return display_inbox(home, caller, render, ref=args[0] if args else None)
    if operation == 'answer':
        ref = args.pop(0) if args and args[0].startswith('Q-') else None
        text = ' '.join(args).strip()
        if not text:
            raise ValueError('MAINTENANCE_ANSWER_TEXT_REQUIRED')
        consent = text.lower() in ('同意', '确认', '拒绝', '不同意', 'yes', 'agree', 'no', 'reject')
        binding = answer_binding(home, caller, ref=ref, consent=consent)
        receipt = commit_control(cli, 'typed_clarify_response',
                                 (ref + ' ' + text) if ref else text, binding=binding)
        source = committed_source_for_receipt(home, cli.session_id, receipt)
        if source is None:
            raise ValueError('MAINTENANCE_ANSWER_COMMIT_FAILED')
        render('回答已提交，维护运行时将重验并处理。')
        return source['source_id']
    if operation == 'submit':
        text = ' '.join(args).strip()
        if not text:
            raise ValueError('MAINTENANCE_REQUEST_TEXT_REQUIRED')
        receipt = commit_control(cli, 'slash_queue', text)
        source = committed_source_for_receipt(home, cli.session_id, receipt)
        if source is None:
            raise ValueError('MAINTENANCE_REQUEST_COMMIT_FAILED')
        render('维护来源已持久交接；关闭 CLI 不取消任务。')
        return source['source_id']
    raise ValueError('Usage: /maintenance inbox | status [task] | answer <ref> <text> | result [task] | cancel <task> | submit text')


def notify_pending(cli):
    if not inbox_enabled(get_hermes_home()):
        return
    home, caller = _caller(cli)
    rows = [r for r in read_inbox(home, caller) if r['kind'] == 'QUESTION' or r['evidence_level'] != 'DISPLAYED']
    if rows:
        cli._console_print(f'维护收件箱有 {len(rows)} 项待处理内容。输入 /maintenance inbox 查看。')


def consume_bare_consent(cli, raw):
    """Only human terminal consent to a unique, locally displayed session prompt."""
    if raw.strip().lower() not in ('同意', '确认', '拒绝', '不同意', 'yes', 'agree', 'no', 'reject'):
        return False
    if not inbox_enabled(get_hermes_home()):
        return False
    try:
        handle_command(cli, shlex.join(['/maintenance', 'answer', raw]))
    except (ValueError, OSError) as exc:
        cli._console_print(str(exc))
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('operation', choices=('submit', 'inbox', 'status', 'answer', 'result', 'cancel'))
    parser.add_argument('arguments', nargs='*')
    args = parser.parse_args(argv)
    from hermes_state import SessionDB
    db = SessionDB(db_path=get_hermes_home() / 'state.db')
    cli = SimpleNamespace(session_id=args.session, _session_db=db,
                          _console_print=lambda text: print(text, flush=True))
    try:
        command = shlex.join(['/maintenance', args.operation, *args.arguments])
        result = handle_command(cli, command)
        print(json.dumps({'operation': args.operation, 'result': result}, ensure_ascii=False), flush=True)
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc) + '\n')
    finally:
        db.close()


if __name__ == '__main__':
    main()
