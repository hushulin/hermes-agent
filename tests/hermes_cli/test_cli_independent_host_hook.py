"""Real CLI slash dispatch and committed caller source; no model/backend startup."""
import json

from hermes_state import SessionDB
from hermes_maintenance_source import delivery_sources, read_committed_source, control_sources
from _maintenance_intake import MaintenanceIntake


def test_real_cli_dispatch_persists_request_and_local_read_is_not_a_task(tmp_path, monkeypatch):
    from cli import HermesCLI
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    (home / 'maintenance-runtime.json').write_text(json.dumps({'enabled': True, 'independent_host': True}), encoding='utf-8')
    (home / 'mem0.json').write_text(json.dumps({'governed': {'maintenance_controls': True,
        'maintenance_intake': True, 'production': {'owners': [
            {'platform': 'cli', 'actor_id': 'local-user', 'memory_user_id': 'owner-1'}]}}}), encoding='utf-8')
    db = SessionDB(db_path=home / 'state.db')
    terminal = object.__new__(HermesCLI)
    terminal.session_id, terminal._session_db = 'cli-session', db
    printed = []
    terminal._console_print = printed.append  # terminal rendering boundary only
    try:
        assert terminal.process_command('/maintenance submit "Correct project database to PostgreSQL"') is True
        source_id, = delivery_sources(home)
        source = read_committed_source(home, source_id)
        assert source['raw_text'] == 'Correct project database to PostgreSQL'
        assert source['authority'] == 'cli-local' and source['control_kind'] == 'slash_queue'
        intake = MaintenanceIntake(home, 'owner-1', {('cli', 'local-user'): 'owner-1'}, enabled=True)
        accepted, = intake.pump()
        assert accepted['source_id'] == source_id
        assert terminal.process_command('/maintenance inbox') is True
        assert delivery_sources(home) == [] and control_sources(home) == []
        with intake._journal().transaction() as journal:
            assert journal.execute('SELECT count(*) FROM maintenance_tasks').fetchone()[0] == 1
    finally:
        db.close()
