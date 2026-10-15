"""Non-executing compatibility notice for the retired private maintenance host.

Keep the former module entry point explicit for old shell commands. This module
must not open state, load plugins, submit work, or interpret an answer as consent.
"""

import sys

RETIRED_MESSAGE = (
    "MAINTENANCE_RETIRED: 旧 core 维护入口已退役；本次命令未执行，"
    "未提交、回答或取消任何任务。已有维护数据保留。"
    "请使用当前独立插件提供的入口。"
)


def handle_command(cli, command, *, display=None):
    """Consume an old slash command with a notice, without executing it."""
    (display or cli._console_print)(RETIRED_MESSAGE)


def main(argv=None):
    """Reject every legacy invocation, including malformed or stale arguments."""
    print(RETIRED_MESSAGE, file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
