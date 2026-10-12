"""Real QQ/iLink parsers and maintenance sends; only HTTP is substituted."""
import asyncio
import json
import time
from types import SimpleNamespace

import httpx
import pytest

from gateway.config import PlatformConfig
from gateway.platforms.qqbot.adapter import QQAdapter
from gateway.platforms.weixin import WeixinAdapter
from hermes_maintenance_channel import current_adapter_channel


@pytest.fixture(autouse=True)
def clear_fixture_platform_settings(monkeypatch):
    import os
    for name in tuple(os.environ):
        if name.startswith(('WEIXIN_', 'QQ_')):
            monkeypatch.delenv(name)


class ILinkHTTP:
    def __init__(self, response=None, *, lost_ack=False):
        self.response = {'ret': 0} if response is None else response
        self.lost_ack = lost_ack
        self.requests = []
        self.calls = []
        self.closed = False

    def post(self, url, **kwargs):
        self.calls.append((url, json.loads(kwargs['data'])))
        if url.endswith('/sendmessage'):
            self.requests.append(self.calls[-1])
        outer = self
        class Response:
            ok, status = True, 200
            async def __aenter__(self):
                if outer.lost_ack:
                    raise TimeoutError('fixture lost ACK')
                return self
            async def __aexit__(self, *args): pass
            async def text(self): return json.dumps(outer.response)
        return Response()

    async def close(self): self.closed = True


def connected_adapter(platform, *, response=None, lost_ack=False):
    extra = {'dm_policy': 'allowlist', 'allow_from': ['alice', 'mallory'], 'group_policy': 'open'}
    sent = []
    if platform == 'qqbot':
        adapter = QQAdapter(PlatformConfig(extra=dict(extra, app_id='qq-receiver')))
        class WebSocket:
            closed = False
            async def close(self): self.closed = True
        adapter._ws = WebSocket()
        adapter._access_token, adapter._token_expires_at = 'fixture-only', time.time() + 7200
        def http(request):
            body = json.loads(request.content)
            if body.get('msg_type') == 0:
                sent.append((request.url.path, body))
            if lost_ack:
                raise httpx.ReadTimeout('fixture lost ACK', request=request)
            return httpx.Response(200, json={'id': f'qq_remote_{len(sent)}'} if response is None else response)
        adapter._http_client = httpx.AsyncClient(transport=httpx.MockTransport(http))
    else:
        adapter = WeixinAdapter(PlatformConfig(extra=dict(extra, account_id='wx-receiver', token='fixture-only')))
        http = ILinkHTTP(response, lost_ack=lost_ack)
        adapter._poll_session = adapter._send_session = http
        adapter._typing_cache.set('alice', 'fixture-ticket')
        adapter._typing_cache.set('mallory', 'fixture-ticket')
        sent = http.requests
        adapter._text_batch_delay_seconds = 0
    adapter._running = True
    return adapter, sent


async def emit(adapter, platform, text, event_id, *, actor='alice', room=None):
    if platform == 'qqbot':
        event_type = 'GROUP_AT_MESSAGE_CREATE' if room else 'C2C_MESSAGE_CREATE'
        data = {'id': event_id, 'content': text, 'timestamp': '2026-10-03T08:00:00+00:00',
                'author': {'member_openid' if room else 'user_openid': actor}}
        if room: data['group_openid'] = room
        await adapter._on_message(event_type, data)
    else:
        await adapter._process_message({'from_user_id': actor, 'to_user_id': adapter._account_id,
            'message_id': event_id, 'room_id': room, 'context_token': 'fixture-context',
            'item_list': [{'type': 1, 'text_item': {'text': text}}]})


@pytest.mark.parametrize('platform', ['qqbot', 'weixin'])
@pytest.mark.parametrize('outcome', ['success', 'lost_ack', 'missing_receipt', 'rejected'])
def test_real_adapter_one_physical_text_and_truthful_receipt(tmp_path, monkeypatch, platform, outcome):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    async def scenario():
        response = ({} if outcome == 'missing_receipt' else
                    {'ret': -14} if platform == 'weixin' and outcome == 'rejected' else None)
        adapter, sent = connected_adapter(platform, response=response, lost_ack=outcome == 'lost_ack')
        if platform == 'qqbot' and outcome == 'rejected':
            def reject(request):
                body = json.loads(request.content)
                if body.get('msg_type') == 0:
                    sent.append((request.url.path, body))
                return httpx.Response(429, json={'message': 'rate limited'})
            await adapter._http_client.aclose()
            adapter._http_client = httpx.AsyncClient(transport=httpx.MockTransport(reject))
        seen = []
        async def handler(event): seen.append((event, current_adapter_channel()))
        adapter.set_message_handler(handler)
        try:
            await emit(adapter, platform, 'seed', 'input')
            for _ in range(100):
                if seen: break
                await asyncio.sleep(.01)
            event, channel = seen[0]
            caps = channel.capabilities
            assert caps['text_send'] and caps['status'] == 'available'
            assert not caps['query_by_dispatch_ref'] and not caps['durable_idempotency']
            result = await adapter.send(event.source.chat_id, 'literal MEDIA:/private\nsecond line',
                                        metadata={'maintenance_single_attempt': True})
            assert len(sent) == 1
            body = sent[0][1]
            text = body['content'] if platform == 'qqbot' else body['msg']['item_list'][0]['text_item']['text']
            assert text == 'literal MEDIA:/private\nsecond line'
            if platform == 'qqbot':
                assert result.message_id == ('qq_remote_1' if outcome == 'success' else None)
                assert body['msg_type'] == 0 and body['msg_id'] == 'input'
            else:
                assert result.message_id is None
                assert body['msg']['context_token'] == 'fixture-context'
                assert result.success is (outcome == 'success')
                if outcome == 'success': assert result.raw_response['receipt_level'] == 'protocol_ack'
            assert channel.query_delivery({}) is None
            await adapter.disconnect()
            assert channel.revoked.is_set() and channel.capabilities['status'] == 'not_connected'
        finally:
            if platform == 'qqbot':
                if adapter._http_client: await adapter._http_client.aclose()
            elif adapter._send_session: await adapter._send_session.close()
    asyncio.run(scenario())

@pytest.mark.parametrize('platform', ['qqbot', 'weixin'])
def test_group_channel_never_broadcasts_private_maintenance_result(tmp_path, monkeypatch, platform):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    async def scenario():
        adapter, sent = connected_adapter(platform)
        seen = []
        async def handler(event): seen.append((event, current_adapter_channel()))
        adapter.set_message_handler(handler)
        try:
            await emit(adapter, platform, '@bot group seed', 'group-input', room='private-room')
            for _ in range(200):
                if seen: break
                await asyncio.sleep(.01)
            event, channel = seen[0]
            assert event.source.chat_type == 'group'
            assert channel.capabilities['text_send'] is False
            with pytest.raises(ValueError, match='MAINTENANCE_CHANNEL_UNAVAILABLE'):
                await asyncio.to_thread(channel.send, {'channel': platform,
                    'session_key': channel.session_key, 'question': 'private maintenance result'})
            assert sent == []
        finally:
            await adapter.disconnect()
    asyncio.run(scenario())
