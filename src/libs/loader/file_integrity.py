"""File integrity checker with SQLite persistence."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path


class FileIntegrityChecker(ABC):
    """Abstract contract for ingestion file integrity checks."""

    @abstractmethod
    def compute_sha256(self, file_path: str) -> str:
        pass

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        pass

    @abstractmethod
    def mark_success(self, file_hash: str, file_path: str, collection: str | None = None) -> None:
        pass

    @abstractmethod
    def mark_failed(self, file_hash: str, file_path: str, error_msg: str) -> None:
        pass


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """SQLite-backed integrity checker with WAL mode."""

    def __init__(self, db_path: str = "data/db/ingestion_history.db") -> None:
        self.db_path = str(db_path)
        path_obj = Path(self.db_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            return conn
        except Exception:
            conn.close()
            raise

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_history (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    status TEXT NOT NULL CHECK(status IN ('success', 'failed', 'processing')),
                    collection TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    error_msg TEXT,
                    chunk_count INTEGER
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ingestion_status ON ingestion_history(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ingestion_processed_at ON ingestion_history(processed_at)"
            )
            conn.commit()
        finally:
            conn.close()

    def _ensure_writable(self) -> None:
        path_obj = Path(self.db_path)
        if path_obj.exists() and not os.access(path_obj, os.W_OK):
            raise RuntimeError("Failed to mark success: database is not writable")

    def compute_sha256(self, file_path: str) -> str:
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path_obj.is_file():
            raise IOError(f"Path is not a file: {file_path}")

        hasher = hashlib.sha256()
        with path_obj.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    def should_skip(self, file_hash: str) -> bool:
        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT status FROM ingestion_history WHERE file_hash = ?",
                (file_hash,),
            )
            row = cursor.fetchone()
            return bool(row and row[0] == "success")
        finally:
            conn.close()

    def mark_success(self, file_hash: str, file_path: str, collection: str | None = None) -> None:
        self._ensure_writable()
        path_obj = Path(file_path)
        file_size = path_obj.stat().st_size if path_obj.exists() and path_obj.is_file() else None
        conn: sqlite3.Connection | None = None
        try:
            conn = self._connect()
            conn.execute(
                """
                INSERT INTO ingestion_history (
                    file_hash, file_path, file_size, status, collection, processed_at, updated_at, error_msg
                ) VALUES (?, ?, ?, 'success', ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, NULL)
                ON CONFLICT(file_hash) DO UPDATE SET
                    file_path=excluded.file_path,
                    file_size=excluded.file_size,
                    status='success',
                    collection=excluded.collection,
                    processed_at=CURRENT_TIMESTAMP,
                    updated_at=CURRENT_TIMESTAMP,
                    error_msg=NULL
                """,
                (file_hash, file_path, file_size, collection),
            )
            conn.commit()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Failed to mark success: {e}") from e
        finally:
            if conn is not None:
                conn.close()

    def mark_failed(self, file_hash: str, file_path: str, error_msg: str) -> None:
        self._ensure_writable()
        path_obj = Path(file_path)
        file_size = path_obj.stat().st_size if path_obj.exists() and path_obj.is_file() else None
        conn: sqlite3.Connection | None = None
        try:
            conn = self._connect()
            conn.execute(
                """
                INSERT INTO ingestion_history (
                    file_hash, file_path, file_size, status, collection, processed_at, updated_at, error_msg
                ) VALUES (?, ?, ?, 'failed', NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
                ON CONFLICT(file_hash) DO UPDATE SET
                    file_path=excluded.file_path,
                    file_size=excluded.file_size,
                    status='failed',
                    processed_at=CURRENT_TIMESTAMP,
                    updated_at=CURRENT_TIMESTAMP,
                    error_msg=excluded.error_msg
                """,
                (file_hash, file_path, file_size, error_msg),
            )
            conn.commit()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Failed to mark failed: {e}") from e
        finally:
            if conn is not None:
                conn.close()

    def close(self) -> None:
        return None
