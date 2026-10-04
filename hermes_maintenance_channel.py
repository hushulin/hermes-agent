"""Borrowed current-adapter capabilities. No services, routes, or credentials."""
from contextlib import contextmanager
from contextvars import ContextVar
import asyncio
import threading
import time

_current = ContextVar('maintenance_channel', default=None)
_SEAL = object()
_RECEIVER_FIELDS = {'feishu': '_bot_open_id', 'qqbot': '_app_id', 'weixin': '_account_id'}


class ProtocolAcknowledgedWithoutMessageRef(ValueError):
    """Remote success ACK, without the message reference required by this outbox."""


class AdapterChannel:
    def __init__(self, adapter, event, session_key, *, _seal):
        if _seal is not _SEAL:
            raise ValueError('TRUSTED_ADAPTER_CHANNEL_REQUIRED')
        self.adapter, self.loop = adapter, asyncio.get_running_loop()
        self.platform = event.source.platform.value
        self.session_key, self.chat_id = session_key, event.source.chat_id
        self.actor = event.source.user_id
        self.receiver = getattr(adapter, _RECEIVER_FIELDS[self.platform], None)
        self.chat_type = event.source.chat_type
        self.thread_id = getattr(event.source, 'thread_id', None)
        self.revoked = threading.Event()

    @property
    def capabilities(self):
        if self.platform == 'feishu':
            transport = getattr(self.adapter, '_client', None) is not None
        elif self.platform == 'qqbot':
            transport = self.adapter.is_connected and self.adapter._http_client is not None
        else:
            session = self.adapter._send_session
            transport = bool(self.adapter._running and session and not session.closed and self.adapter._token)
        text_send = self.chat_type == 'dm' and (self.platform != 'qqbot'
                    or self.adapter._chat_type_map.get(self.chat_id) == 'c2c')
        connected = (transport and text_send and bool(self.receiver) and not self.revoked.is_set()
                     and self.loop.is_running()
                     and getattr(self.adapter, _RECEIVER_FIELDS[self.platform], None) == self.receiver)
        return {'platform': self.platform, 'text_send': text_send,
                'query_by_dispatch_ref': False, 'durable_idempotency': False,
                'receipt_level': 'protocol_ack' if self.platform == 'weixin' else 'remote_message_id',
                'status': 'available' if connected else 'not_connected'}

    def matches(self, source):
        namespace = (f'feishu/{self.receiver}/{self.session_key}' if self.platform == 'feishu'
                     else f'{self.platform}-human/{self.receiver}/{self.chat_type}/{self.chat_id}')
        return (self.capabilities['status'] == 'available'
                and source.get('authority') == self.platform + '-human'
                and source.get('session_key') == self.session_key
                and source.get('actor') == self.actor
                and source.get('audience') == 'private'
                and source.get('namespace') == namespace
                and source.get('receiver', self.receiver) == self.receiver
                and source.get('chat_type', self.chat_type) == self.chat_type
                and (source.get('thread_id') or '') == (self.thread_id or '')
                # The original sealed plain-text contract stores the recipient
                # in its trusted session key/namespace, without a chat_id column.
                and source.get('chat_id', self.chat_id) == self.chat_id)

    def send(self, payload):
        if (self.capabilities['status'] != 'available'
                or payload.get('channel') != self.platform
                or payload.get('session_key') != self.session_key):
            raise ValueError('MAINTENANCE_CHANNEL_UNAVAILABLE')
        try:
            if asyncio.get_running_loop() is self.loop:
                raise ValueError('MAINTENANCE_SEND_REQUIRES_WORKER_THREAD')
        except RuntimeError:
            pass
        async def dispatch():
            if self.capabilities['status'] != 'available':
                raise ValueError('MAINTENANCE_CHANNEL_UNAVAILABLE')
            return await self.adapter.send(self.chat_id, payload['question'], metadata={
                'maintenance_single_attempt': True, 'thread_id': self.thread_id})
        future = asyncio.run_coroutine_threadsafe(dispatch(), self.loop)
        try:
            result = future.result(timeout=10)
        except BaseException:
            future.cancel()  # A remote send may already have happened: outbox stays UNCERTAIN.
            raise
        if result.success and self.capabilities['receipt_level'] == 'protocol_ack':
            return {'evidence_level': 'API_ACCEPTED', 'message_ref': None,
                    'channel': self.platform, 'session_key': self.session_key,
                    'accepted_at': time.time()}
        if not result.success or not isinstance(result.message_id, str) or not result.message_id:
            raise ValueError('PLATFORM_MESSAGE_REF_UNAVAILABLE')
        return {'message_ref': result.message_id, 'channel': self.platform,
                'session_key': self.session_key, 'delivered_at': time.time()}

    def query_delivery(self, payload):
        # Feishu get(message_id) cannot discover a lost create ACK by dispatch ref.
        return None


@contextmanager
def adapter_channel_scope(adapter, event, session_key):
    platform = getattr(getattr(event.source, 'platform', None), 'value', None)
    if platform not in ('feishu', 'qqbot', 'weixin'):
        token = _current.set(None)
        try:
            yield None
        finally:
            _current.reset(token)
        return
    channels = getattr(adapter, '_maintenance_channels', None)
    if channels is None:
        channels = adapter._maintenance_channels = {}
    key = (platform, session_key, event.source.user_id,
           getattr(adapter, _RECEIVER_FIELDS[platform], None), event.source.chat_id,
           getattr(event.source, 'thread_id', None))
    channel = channels.get(key)
    if channel is None or channel.revoked.is_set() or channel.loop is not asyncio.get_running_loop():
        if len(channels) >= 64:
            # Ordinary messaging continues. A maintenance host needs a channel
            # explicitly present in this bounded registry to drive delivery.
            token = _current.set(None)
            try:
                yield None
            finally:
                _current.reset(token)
            return
        channel = channels[key] = AdapterChannel(adapter, event, session_key, _seal=_SEAL)
    token = _current.set(channel)
    try:
        yield channel
    finally:
        _current.reset(token)


def current_adapter_channel():
    return _current.get()


def revoke_adapter_channels(adapter):
    for channel in getattr(adapter, '_maintenance_channels', {}).values():
        channel.revoked.set()
    getattr(adapter, '_maintenance_channels', {}).clear()
