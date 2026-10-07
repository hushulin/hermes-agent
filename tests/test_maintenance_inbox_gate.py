import json

import pytest

from hermes_maintenance_inbox import inbox_enabled


@pytest.mark.parametrize('runtime,expected', [
    ({'enabled': True, 'mode': 'production', 'independent_host': False}, True),
    ({'enabled': True, 'mode': 'production', 'independent_host': True}, True),
    ({'enabled': True, 'mode': 'isolated-test', 'independent_host': False}, True),
    ({'enabled': False, 'mode': 'production', 'independent_host': False}, False),
    ({'enabled': True, 'mode': 'shadow', 'independent_host': False}, False),
    ({'enabled': True, 'mode': 'production', 'independent_host': 'false'}, False),
])
def test_owner_inbox_gate_matches_enabled_runtime_mode(tmp_path, runtime, expected):
    (tmp_path / 'maintenance-runtime.json').write_text(json.dumps(runtime), encoding='utf-8')
    assert inbox_enabled(tmp_path) is expected


def test_owner_inbox_gate_fails_closed_for_missing_or_invalid_config(tmp_path):
    assert inbox_enabled(tmp_path) is False
    path = tmp_path / 'maintenance-runtime.json'
    path.write_text('{', encoding='utf-8')
    assert inbox_enabled(tmp_path) is False
    path.write_text('[]', encoding='utf-8')
    assert inbox_enabled(tmp_path) is False
