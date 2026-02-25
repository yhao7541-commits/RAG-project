"""MCP protocol-aligned error helpers."""

from __future__ import annotations


class MCPError(Exception):
    def __init__(self, code: int, message: str, data: dict[str, object] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data or {}

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "data": self.data}


PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
