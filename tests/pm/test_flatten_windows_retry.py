"""Windows extraction can briefly hold a directory open after unpacking."""
import errno
from pathlib import Path

import pytest

from pm.store import flatten_single_dir

pytestmark = pytest.mark.platforms("windows")


@pytest.fixture(autouse=True)
def instant_retry(monkeypatch):
    monkeypatch.setattr('pm.filesystem.time.sleep', lambda delay: None)


def test_flatten_recovers_from_transient_hold(tmp_path, monkeypatch):
    inner = tmp_path / "archive"
    (inner / "bin").mkdir(parents=True)
    (inner / "bin" / "tool.exe").write_bytes(b"verified payload")
    original = Path.rename
    attempts = []

    def rename(path, target):
        attempts.append(path)
        if len(attempts) < 3:
            error = PermissionError(errno.EACCES, "temporary hold")
            error.winerror = 5
            raise error
        return original(path, target)

    monkeypatch.setattr(Path, "rename", rename)
    flatten_single_dir(tmp_path)
    assert (tmp_path / "bin" / "tool.exe").read_bytes() == b"verified payload"
    assert not inner.exists()
    assert len(attempts) == 3


@pytest.mark.parametrize("error_number,winerror,retryable", [
    (errno.EACCES, 5, True), (errno.EINVAL, 87, False),
])
def test_flatten_preserves_payload_when_move_fails(tmp_path, monkeypatch, error_number, winerror, retryable):
    inner = tmp_path / "archive"
    inner.mkdir()
    payload = inner / "payload"
    payload.write_bytes(b"verified payload")
    attempts = []

    def rename(path, target):
        attempts.append(path)
        error = OSError(error_number, "move failed")
        error.winerror = winerror
        raise error

    monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises(OSError):
        flatten_single_dir(tmp_path)
    assert payload.read_bytes() == b"verified payload"
    assert (1 < len(attempts) < 100) if retryable else len(attempts) == 1


def test_flatten_retries_empty_directory_removal(tmp_path, monkeypatch):
    inner = tmp_path / 'archive'
    inner.mkdir()
    (inner / 'payload').write_bytes(b'verified payload')
    original = Path.rmdir
    attempts = []

    def rmdir(path):
        attempts.append(path)
        if len(attempts) == 1:
            raise PermissionError(errno.EACCES, 'directory temporarily held')
        return original(path)

    monkeypatch.setattr(Path, 'rmdir', rmdir)
    flatten_single_dir(tmp_path)
    assert (tmp_path / 'payload').read_bytes() == b'verified payload'
    assert not inner.exists()
    assert len(attempts) == 2
