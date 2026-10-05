"""
관리 콘솔 영속 저장 (SQLite, admin.db) — 저장된 탐색과 같은 파일, 다른 표.

여기 담기는 것은 전부 **운영자 UI 상태·프로젝트 메타**이지 지식(그래프)이
아니다. 지식은 KG 체크포인트에, 이것은 admin.db 에. 둘을 섞지 않는다.
stdlib sqlite3 만 쓰므로 커널 의존성은 늘지 않는다.

- SettingsStore: 전역 키-값 (LLM provider/model/base_url 등). **API 키는 절대
  담지 않는다** — 키는 환경변수에만 있고, 여기엔 "어떤 provider/model 인가"만.
- ProjectStore: 네임스페이스=프로젝트 메타 (빈 프로젝트도 목록에 남게).
- SchemaDeclStore: 선언적 스키마 주석 (술어 domain/range·타입 설명) — 관측
  스키마 위에 얹는 "의도". lint 이 이걸로 domain/range 위반을 잡는다.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class _Db:
    """admin.db 커넥션 헬퍼 — 각 스토어가 상속받아 같은 파일을 공유."""

    def __init__(self, path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:  # 하위 클래스가 구현
        raise NotImplementedError


class SettingsStore(_Db):
    """전역 키-값 설정. 값은 JSON 직렬화 (문자열·객체 모두 수용)."""

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY, value TEXT NOT NULL)""")

    def get(self, key: str, default=None):
        with self._conn() as c:
            row = c.execute("SELECT value FROM settings WHERE key = ?",
                            (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key: str, value) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO settings VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value, ensure_ascii=False)))


class ProjectStore(_Db):
    """네임스페이스=프로젝트 메타. 노드 0개인 빈 프로젝트도 여기 있으면
    목록에 뜬다 (그래프 파생 목록만으로는 빈 프로젝트가 안 보인다)."""

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS projects (
                namespace   TEXT PRIMARY KEY,
                description TEXT NOT NULL DEFAULT '',
                domain      TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL)""")

    @staticmethod
    def _row(r: sqlite3.Row) -> Dict[str, Any]:
        return {"namespace": r["namespace"], "description": r["description"],
                "domain": r["domain"], "created_at": r["created_at"]}

    def list(self) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM projects ORDER BY created_at DESC").fetchall()
        return [self._row(r) for r in rows]

    def get(self, namespace: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM projects WHERE namespace = ?",
                          (namespace,)).fetchone()
        return self._row(r) if r else None

    def create(self, namespace: str, description: str = "",
               domain: str = "") -> Dict[str, Any]:
        created_at = datetime.now().isoformat()
        try:
            with self._conn() as c:
                c.execute("INSERT INTO projects VALUES (?, ?, ?, ?)",
                          (namespace, description or "", domain or "", created_at))
        except sqlite3.IntegrityError:
            return {"error": "duplicate", "detail": namespace}
        logger.info(f"📁 Project record created: {namespace}")
        return {"namespace": namespace, "description": description or "",
                "domain": domain or "", "created_at": created_at}

    def delete(self, namespace: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM projects WHERE namespace = ?", (namespace,))


class SchemaDeclStore(_Db):
    """선언적 스키마 주석 — 관측 스키마 위에 얹는 '의도'.

    - predicate 선언: (namespace, predicate) → {domain, range, description}
    - type 선언: (namespace, type) → {description, deprecated}
    관측(instances)은 그래프가, 선언은 여기가 원본. lint 이 둘을 대조한다.
    """

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS predicate_decl (
                namespace TEXT NOT NULL, predicate TEXT NOT NULL,
                domain TEXT NOT NULL DEFAULT '', range TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (namespace, predicate))""")
            c.execute("""CREATE TABLE IF NOT EXISTS type_decl (
                namespace TEXT NOT NULL, type TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                deprecated INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (namespace, type))""")

    def predicates(self, namespace: str) -> Dict[str, Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM predicate_decl WHERE namespace = ?",
                (namespace,)).fetchall()
        return {r["predicate"]: {"domain": r["domain"], "range": r["range"],
                                 "description": r["description"]} for r in rows}

    def set_predicate(self, namespace: str, predicate: str, domain: str = "",
                      range_: str = "", description: str = "") -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO predicate_decl VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(namespace, predicate) DO UPDATE SET "
                "domain = excluded.domain, range = excluded.range, "
                "description = excluded.description",
                (namespace, predicate, domain or "", range_ or "",
                 description or ""))

    def types(self, namespace: str) -> Dict[str, Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM type_decl WHERE namespace = ?",
                (namespace,)).fetchall()
        return {r["type"]: {"description": r["description"],
                            "deprecated": bool(r["deprecated"])} for r in rows}

    def set_type(self, namespace: str, type_: str, description: str = "",
                 deprecated: bool = False) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO type_decl VALUES (?, ?, ?, ?) "
                "ON CONFLICT(namespace, type) DO UPDATE SET "
                "description = excluded.description, deprecated = excluded.deprecated",
                (namespace, type_, description or "", 1 if deprecated else 0))

    def rename_type(self, namespace: str, old: str, new: str) -> None:
        """타입 개명 시 선언도 따라간다 (없으면 무해)."""
        with self._conn() as c:
            c.execute("UPDATE OR IGNORE type_decl SET type = ? "
                      "WHERE namespace = ? AND type = ?", (new, namespace, old))
            c.execute("DELETE FROM type_decl WHERE namespace = ? AND type = ? "
                      "AND EXISTS (SELECT 1 FROM type_decl WHERE namespace = ? "
                      "AND type = ?)", (namespace, old, namespace, new))

    def delete_namespace(self, namespace: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM predicate_decl WHERE namespace = ?", (namespace,))
            c.execute("DELETE FROM type_decl WHERE namespace = ?", (namespace,))
