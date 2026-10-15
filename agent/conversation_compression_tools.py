"""Refresh the tool surface only at a permitted compaction commit boundary."""

import logging

logger = logging.getLogger(__name__)


def refresh_agent_tool_definitions(agent) -> bool:
    """Rebuild agent.tools at the compaction commit boundary (the only moment config reaches a forever-session's
    frozen tool schemas; the prompt cache is already invalid). Returns True when tools were added."""
    if getattr(agent, "_skip_mcp_refresh", False):
        # In-memory review forks keep an instance-local tool surface. The between-turns
        # hook honors the same opt-out; rebuilding here would replace their schemas.
        return False
    from tools.mcp_tool_agent import refresh_agent_mcp_tools
    added = refresh_agent_mcp_tools(agent, content_aware=True)
    if added:
        logger.info("Compaction tool refresh added tools: %s", sorted(added))
    return bool(added)
