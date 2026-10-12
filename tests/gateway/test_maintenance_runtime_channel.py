"""Current-adapter handoff and actual Feishu text send at the network boundary."""
import asyncio
import json
from types import SimpleNamespace
import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from plugins.platforms.feishu.adapter import FeishuAdapter
from hermes_maintenance_channel import (AdapterChannel, current_adapter_channel,
                                       revoke_adapter_channels)
from agent.agent_init import _memory_provider_init_kwargs, _GATEWAY_IDENTITY_PARAMS


def test_base_handler_passes_borrowed_current_channel_to_provider_kwargs(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    async def scenario():
        adapter = FeishuAdapter(PlatformConfig(extra={}))
        adapter._bot_open_id = 'receiver-bot'
        adapter._client = object()  # Connected network boundary; no calls in this test.
        source = adapter.build_source(chat_id='oc_fixture', chat_type='dm', user_id='alice')
        event = MessageEvent(text='hello', source=source, message_type=MessageType.TEXT, message_id='om_input')
        observed = []
        agent = SimpleNamespace(session_id='s1', _session_db=None, session_cwd=None,
                                **{'_' + k: None for k in _GATEWAY_IDENTITY_PARAMS})
        async def handler(event):
            kwargs = _memory_provider_init_kwargs(agent, 'feishu')
            channel = kwargs['maintenance_channel']
            observed.append(channel)
            assert channel.adapter is adapter
            assert channel.capabilities == {'platform': 'feishu', 'text_send': True,
                'query_by_dispatch_ref': False, 'durable_idempotency': False,
                'receipt_level': 'remote_message_id', 'status': 'available'}
            assert channel.query_delivery({}) is None
            return 'handled'
        adapter.set_message_handler(handler)
        assert await adapter._call_message_handler(event) == 'handled'
        assert await adapter._call_message_handler(event) == 'handled'
        assert observed[0] is observed[1]
        assert current_adapter_channel() is None
        assert _memory_provider_init_kwargs(agent, 'feishu')['maintenance_channel'] is None
        revoke_adapter_channels(adapter)
        assert observed[0].revoked.is_set()
        assert adapter._maintenance_channels == {}
    asyncio.run(scenario())


@pytest.mark.parametrize('failure', ['lost_ack', 'missing_id', 'success'])
def test_feishu_maintenance_send_is_one_actual_create_and_platform_ref(tmp_path, monkeypatch, failure):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    async def scenario():
        adapter = FeishuAdapter(PlatformConfig(extra={}))
        requests = []
        def network(request):
            requests.append(request)
            if failure in ('lost_ack', 'missing_id'):
                raise TimeoutError('network fixture')
            return SimpleNamespace(success=lambda: True,
                data=SimpleNamespace(message_id='om_platform' if failure == 'success' else None))
        adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(create=network))))
        try:
            result = await adapter.send('oc_fixture', '需要澄清具体目标。',
                                        metadata={'maintenance_single_attempt': True})
            assert len(requests) == 1
            assert requests[0].request_body.msg_type == 'text'
            assert json.loads(requests[0].request_body.content)['text'] == '需要澄清具体目标。'
            assert result.message_id == ('om_platform' if failure == 'success' else None)
            if failure == 'lost_ack':
                assert result.success is False
        finally:
            adapter._shutdown_sdk_executor()
    asyncio.run(scenario())


def test_channel_cannot_be_constructed_without_host_seal():
    with pytest.raises(ValueError, match='TRUSTED_ADAPTER_CHANNEL_REQUIRED'):
        AdapterChannel(None, None, None, _seal=None)
