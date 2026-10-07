"""Behavior contracts for the sandbox proxy's upstream TLS retry boundary."""

import importlib.util
import socket
import ssl
import sys
from pathlib import Path

import pytest


@pytest.fixture
def proxy_module(monkeypatch, tmp_path):
    source = Path(__file__).parents[2] / 'scripts' / 'sandbox' / 'proxy.py'
    monkeypatch.setattr(
        sys,
        'argv',
        ['proxy.py', str(tmp_path / 'fixtures'), str(tmp_path / 'certs'), str(tmp_path / 'ca.pem')],
    )
    spec = importlib.util.spec_from_file_location('sandbox_proxy_under_test', source)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Upstream:
    def __init__(self):
        self.writes = []

    def sendall(self, data):
        self.writes.append(data)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _RawSocket:
    def close(self):
        pass


def test_forward_https_retries_pre_request_tls_eof_then_relays(proxy_module, monkeypatch):
    upstream = _Upstream()
    calls = []

    def connect(*_args, **_kwargs):
        calls.append('connect')
        if len(calls) == 1:
            raise ssl.SSLEOFError(ssl.SSL_ERROR_EOF, 'eof')
        return _RawSocket()

    class FakeContext:
        def wrap_socket(self, _raw, **_kwargs):
            return upstream

    monkeypatch.setattr(proxy_module.socket, 'create_connection', connect)
    monkeypatch.setattr(proxy_module.ssl, 'create_default_context', lambda **_kwargs: FakeContext())
    monkeypatch.setattr(proxy_module, 'relay', lambda source, destination: calls.append('relay'))
    monkeypatch.setattr(proxy_module.time, 'sleep', lambda delay: calls.append(delay))

    proxy_module.forward_https(object(), 'registry.npmjs.org', 443, b'GET / HTTP/1.1\r\n\r\n')

    assert calls == ['connect', 0.25, 'connect', 'relay']
    assert len(upstream.writes) == 1


def test_forward_https_does_not_retry_certificate_validation_errors(proxy_module, monkeypatch):
    calls = []

    class FakeContext:
        def wrap_socket(self, _raw, **_kwargs):
            raise ssl.SSLCertVerificationError(1, 'certificate failure')

    monkeypatch.setattr(proxy_module.socket, 'create_connection', lambda *_args, **_kwargs: _RawSocket())
    monkeypatch.setattr(proxy_module.ssl, 'create_default_context', lambda **_kwargs: FakeContext())
    monkeypatch.setattr(proxy_module.time, 'sleep', lambda delay: calls.append(delay))

    with pytest.raises(ssl.SSLCertVerificationError):
        proxy_module.forward_https(object(), 'registry.npmjs.org', 443, b'GET / HTTP/1.1\r\n\r\n')

    assert calls == []


def test_forward_https_does_not_retry_after_request_bytes_are_sent(proxy_module, monkeypatch):
    calls = []

    class FailingUpstream(_Upstream):
        def sendall(self, data):
            super().sendall(data)
            raise ssl.SSLEOFError(ssl.SSL_ERROR_EOF, 'eof after request')

    upstream = FailingUpstream()

    class FakeContext:
        def wrap_socket(self, _raw, **_kwargs):
            return upstream

    def connect(*_args, **_kwargs):
        calls.append('connect')
        return _RawSocket()

    monkeypatch.setattr(proxy_module.socket, 'create_connection', connect)
    monkeypatch.setattr(proxy_module.ssl, 'create_default_context', lambda **_kwargs: FakeContext())
    monkeypatch.setattr(proxy_module.time, 'sleep', lambda delay: calls.append(delay))

    with pytest.raises(ssl.SSLEOFError):
        proxy_module.forward_https(object(), 'registry.npmjs.org', 443, b'GET / HTTP/1.1\r\n\r\n')

    assert calls == ['connect']