"""Minimal MCP stdio server skeleton with protocol-safe JSON-RPC handling."""

from __future__ import annotations

import json
import sys
from typing import Any, Callable

from src.core.query_engine import RetrievalPipeline
from src.libs.vector_store import VectorStoreFactory
from src.mcp_server.errors import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    MCPError,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
)
from src.mcp_server.tools.get_document_summary import get_document_summary
from src.mcp_server.tools.list_collections import list_collections
from src.mcp_server.tools.query_knowledge_hub import query_knowledge_hub
from src.observability.logger import get_logger


class MCPServer:
    """MCP server over stdio transport (JSON-RPC 2.0)."""

    def __init__(self, settings: Any):
        self.settings = settings
        self.logger = get_logger("mcp-server")
        self.pipeline = RetrievalPipeline(settings)
        self.vector_store = VectorStoreFactory.create(settings)

        self._methods: dict[str, Callable[[dict[str, Any]], Any]] = {
            "initialize": self._initialize,
            "tools/list": self._tools_list,
            "tools/call": self._tools_call,
        }

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "protocolVersion": "2025-06-18",
            "serverInfo": {"name": "modular-rag-mcp-server", "version": "0.1.0"},
            "capabilities": {"tools": {"listChanged": False}},
        }

    def _tools_list(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "tools": [
                {
                    "name": "query_knowledge_hub",
                    "description": "Hybrid retrieve top-k context with citations",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer"},
                            "collection": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "list_collections",
                    "description": "List available vector-store collections",
                    "inputSchema": {"type": "object", "properties": {}},
                },
                {
                    "name": "get_document_summary",
                    "description": "Get summary metadata for a document",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"doc_id": {"type": "string"}},
                        "required": ["doc_id"],
                    },
                },
            ]
        }

    def _tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise MCPError(INVALID_PARAMS, "tools/call.arguments must be an object")

        if name == "query_knowledge_hub":
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                raise MCPError(INVALID_PARAMS, "query must be a non-empty string")
            top_k = arguments.get("top_k")
            if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
                raise MCPError(INVALID_PARAMS, "top_k must be a positive integer")
            collection = arguments.get("collection")
            if collection is not None and not isinstance(collection, str):
                raise MCPError(INVALID_PARAMS, "collection must be a string")
            return query_knowledge_hub(self.pipeline, query, top_k=top_k, collection=collection)

        if name == "list_collections":
            return list_collections(self.vector_store)

        if name == "get_document_summary":
            doc_id = arguments.get("doc_id")
            if not isinstance(doc_id, str) or not doc_id.strip():
                raise MCPError(INVALID_PARAMS, "doc_id must be a non-empty string")
            return get_document_summary(self.vector_store, doc_id)

        raise MCPError(METHOD_NOT_FOUND, f"Unknown tool: {name}")

    def handle_request(self, request_obj: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request_obj, dict):
            raise MCPError(INVALID_REQUEST, "Invalid JSON-RPC request object")

        method = request_obj.get("method")
        if not isinstance(method, str):
            raise MCPError(INVALID_REQUEST, "Missing or invalid method")

        params = request_obj.get("params") or {}
        if not isinstance(params, dict):
            raise MCPError(INVALID_PARAMS, "params must be an object")

        handler = self._methods.get(method)
        if handler is None:
            raise MCPError(METHOD_NOT_FOUND, f"Method not found: {method}")

        return handler(params)

    def serve_stdio(self) -> None:
        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue

            response: dict[str, Any]
            request_id: Any = None
            try:
                request_obj = json.loads(raw)
                request_id = request_obj.get("id") if isinstance(request_obj, dict) else None
                result = self.handle_request(request_obj)
                response = {"jsonrpc": "2.0", "id": request_id, "result": result}
            except json.JSONDecodeError:
                response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": PARSE_ERROR, "message": "Parse error"},
                }
            except MCPError as e:
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": e.to_dict(),
                }
            except Exception as e:  # noqa: BLE001
                self.logger.exception("Unhandled error while serving request")
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": INTERNAL_ERROR, "message": str(e)},
                }

            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()
