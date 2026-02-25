"""Application entrypoint.

Stage A3 wires Settings loading + minimal logger.
"""

from __future__ import annotations

import sys

from src.core.settings import SettingsError, load_settings
from src.mcp_server import MCPServer
from src.observability.logger import get_logger


def main() -> None:
    logger = get_logger("mcp-server")

    try:
        settings = load_settings("config/settings.yaml")
    except SettingsError as e:
        logger.error(str(e))
        raise SystemExit(1) from e

    logger.info(
        "Settings loaded (llm=%s, embedding=%s, vector_store=%s)",
        settings.llm.provider,
        settings.embedding.provider,
        settings.vector_store.provider,
    )

    if "--serve-stdio" in sys.argv:
        logger.info("Starting MCP server (stdio)")
        server = MCPServer(settings)
        server.serve_stdio()
        return

    logger.info("Settings loaded. Use --serve-stdio to start MCP transport loop")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
