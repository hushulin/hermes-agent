"""Legacy maintenance commands report retirement without opening user state."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from hermes_cli.maintenance_host import RETIRED_MESSAGE, handle_command, main


@pytest.mark.parametrize('command', [
    '/maintenance', '/maintenance inbox', '/maintenance status old-task',
    '/maintenance answer Q-old yes', '/maintenance cancel old-task',
    '/maintenance submit do work', '/maintenance answer "unterminated',
])
def test_old_commands_only_report_retirement(command, tmp_path, monkeypatch):
    home = tmp_path / 'profile'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    existing = home / 'state.db'
    existing.write_bytes(b'existing database must not be opened or mutated')
    messages = []
    cli = SimpleNamespace(_console_print=messages.append)
    assert handle_command(cli, command) is None
    assert messages == [RETIRED_MESSAGE]
    assert existing.read_bytes() == b'existing database must not be opened or mutated'
    assert sorted(path.name for path in home.iterdir()) == ['state.db']


def test_legacy_module_cli_exits_nonzero_without_state(tmp_path, monkeypatch):
    home = tmp_path / 'profile'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    result = subprocess.run(
        [sys.executable, '-m', 'hermes_cli.maintenance_host', '--session', 'old', 'submit', 'work'],
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 2
    assert result.stdout == ''
    assert 'MAINTENANCE_RETIRED' in result.stderr
    assert list(home.iterdir()) == []


def test_legacy_main_rejects_malformed_arguments(capsys):
    with pytest.raises(SystemExit) as error:
        main(['--unknown'])
    assert error.value.code == 2
    assert 'MAINTENANCE_RETIRED' in capsys.readouterr().err


def test_registry_reserves_legacy_name_as_nonexecuting_notice():
    from hermes_cli.commands import resolve_command
    definition = resolve_command('maintenance')
    assert definition is not None
    assert 'retired' in definition.description.lower()
    assert not definition.subcommands
