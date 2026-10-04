"""Maintenance controls at the authenticated gateway command boundary."""
import shlex

from hermes_maintenance_inbox import (answer_binding, independent_enabled, read_inbox,
    read_status, format_status, prepare_projection, record_projection, begin_projection_send)
from hermes_maintenance_source import (acknowledge_control_source, commit_platform_control_source,
    derive_platform_control_source, mint_feishu_text, revise_original_input)


async def handle_command(runner, event):
    source = event.source
    if source.platform.value not in ('feishu', 'qqbot', 'weixin'):
        return '此入口不支持维护命令。'
    if source.chat_type != 'dm':
        return '请在与机器人的私聊中输入 /maintenance inbox。'
    home = runner._resolve_profile_home_for_source(source)
    if not independent_enabled(home):
        return 'MAINTENANCE_INDEPENDENT_HOST_DISABLED'
    adapter = runner._intake_adapter_for(source)
    key = runner._session_key_for_source(source)
    base = (mint_feishu_text(event, adapter, home, key) if source.platform.value == 'feishu'
            else getattr(event, '_maintenance_source', None))
    entry = await runner.async_session_store.get_or_create_session(source)

    def commit(kind, text, binding=None):
        receipt = derive_platform_control_source(base, kind=kind, text=text, binding=binding)
        if receipt is not None and kind != 'maintenance_inbox_read':
            # The same event first authenticates the read, then seals its bound
            # mutation as an immutable control revision (never a capture task).
            receipt = revise_original_input(receipt, text)
        sid = commit_platform_control_source(home, entry.session_id, receipt)
        if sid is None:
            raise ValueError('MAINTENANCE_CONTROL_COMMIT_REQUIRED')
        return sid

    try:
        args = shlex.split(event.get_command_args())
        operation = args.pop(0).lower() if args else 'inbox'
        caller = commit('maintenance_inbox_read', event.text)
        # Read controls are never captured as ordinary maintenance work.
        acknowledge_control_source(home, caller, 'inbox-read:' + caller)
        if operation in ('status', 'result'):
            task = args[0] if args else None
            text = format_status(read_status(home, caller, task_id=task))
            if operation == 'status':
                return text
            rows = [r for r in read_inbox(home, caller, task_id=task) if r['kind'] == 'RESULT']
        elif operation == 'inbox':
            rows = read_inbox(home, caller)
            text = '没有待处理维护内容。' if not rows else '维护内容已尝试发送；渠道接受不等于本人已读。'
        elif operation == 'answer':
            if len(args) < 2:
                raise ValueError('Usage: /maintenance answer <ref> <text>')
            ref, answer = args[0], ' '.join(args[1:])
            binding = answer_binding(home, caller, ref=ref, text=answer)
            question = next(r for r in read_inbox(home, caller) if r['ref'] == ref)
            commit('typed_clarify_response', ref + ' ' + answer if question['channel'] == 'local-inbox' else answer, binding)
            return '回答已提交，独立 host 将重验并处理。'
        elif operation == 'cancel':
            if len(args) != 1 or not any(r['task_id'] == args[0] for r in read_status(home, caller, task_id=args[0])):
                raise ValueError('OWNER_TASK_UNAVAILABLE')
            commit('maintenance_cancel', args[0])
            return '取消请求已提交，独立 host 将重验；这不表示运行中的操作已停止。'
        else:
            raise ValueError('Usage: /maintenance inbox | status [task] | answer <ref> <text> | result [task] | cancel <task>')
        for row in rows:
            projection = prepare_projection(home, caller, row)
            if not begin_projection_send(home, caller, row['ref']):
                text += '\n通知证据：' + projection['evidence_level'] + '。'
                continue
            # One physical send. Only the returned API evidence is persisted.
            sent = await adapter.send(source.chat_id, projection['body'], metadata={
                'maintenance_single_attempt': True, 'thread_id': source.thread_id})
            if not record_projection(home, caller, row['ref'], sent):
                text += '\n通知证据：DELIVERY_UNKNOWN。'
            else:
                text += '\n通知证据：' + ('API_ACCEPTED' if source.platform.value == 'weixin' else 'REMOTE_MESSAGE_ID') + '。'
        return text
    except (ValueError, OSError) as exc:
        return str(exc)


async def handle_projected_reply(runner, event):
    """Keep Weixin's strict intent grammar on the ordinary typed-reply route."""
    import re
    from hermes_maintenance_inbox import projected_reply_ref
    source = event.source
    if (source.platform.value != 'weixin' or source.chat_type != 'dm'
            or not re.fullmatch(r'(?:确认|同意|拒绝|不同意) WX-[A-Z2-7]{16}|回答 WX-[A-Z2-7]{16}：[^\r\n]+', (event.text or '').strip())):
        return None
    home = runner._resolve_profile_home_for_source(source)
    if not independent_enabled(home):
        return None
    base = getattr(event, '_maintenance_source', None)
    entry = await runner.async_session_store.get_or_create_session(source)
    try:
        control = derive_platform_control_source(base, kind='maintenance_inbox_read', text=event.text)
        caller = commit_platform_control_source(home, entry.session_id, control)
        if caller is None:
            return None
        acknowledge_control_source(home, caller, 'inbox-read:' + caller)
        event._maintenance_control_read = caller
        ref = projected_reply_ref(home, caller, event.text.strip())
        if ref is None:
            return None
        binding = answer_binding(home, caller, ref=ref, text=event.text.strip())
        control = derive_platform_control_source(base, kind='typed_clarify_response', text=ref + ' ' + event.text.strip(), binding=binding)
        if control is not None:
            control = revise_original_input(control, control.raw)
        if commit_platform_control_source(home, entry.session_id, control) is None:
            return None
        return ''
    except (ValueError, OSError):
        return None
