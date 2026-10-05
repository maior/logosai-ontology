"""
저장된 탐색(Saved Explorations) — 관리 콘솔 운영자 편의 (SQLite 영속).

팔란티어 Object Explorer 의 saved explorations 를 우리 온톨로지에 맞춘 것:
네임스페이스별로 탐색 필터(검색어·클래스·술어·프로퍼티·trust)를 이름 붙여
저장하고 클릭 한 번으로 복원한다.

경계: 이것은 **운영자 UI 상태**이지 지식(그래프)이 아니다 — KG 체크포인트와
섞지 않고 별도 SQLite 파일(admin.db)에 둔다. stdlib sqlite3 만 쓰므로 커널
의존성은 늘지 않는다 (없으면 죽는 게 아니라, 애초에 파이썬 표준이라 항상 있다).
필터는 불투명 JSON blob — 프론트가 무엇을 담든 백엔드는 해석하지 않는다
(스키마 강결합을 피한다: 필터 모양이 바뀌어도 저장소는 그대로다).
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger


class SavedViewStore:
    """한 admin.db 파일의 저장된 탐색 저장소."""

    def __init__(self, path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS saved_views (
                    id         TEXT PRIMARY KEY,
                    namespace  TEXT NOT NULL,
                    name       TEXT NOT NULL,
                    filter     TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )""")
            # 네임스페이스별 조회·삭제가 주 접근 경로다
            conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_saved_views_ns "
                "ON saved_views(namespace)")
            # 같은 네임스페이스 안에서 이름은 유일 (중복 저장 방지)
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_saved_views_ns_name "
                "ON saved_views(namespace, name)")

    @staticmethod
    def _row(r: sqlite3.Row) -> Dict[str, Any]:
        return {"id": r["id"], "namespace": r["namespace"], "name": r["name"],
                "filter": json.loads(r["filter"]), "created_at": r["created_at"]}

    def list(self, namespace: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM saved_views WHERE namespace = ? "
                "ORDER BY created_at DESC", (namespace,)).fetchall()
        return [self._row(r) for r in rows]

    def create(self, namespace: str, name: str,
               filter_: Dict[str, Any]) -> Dict[str, Any]:
        name = (name or "").strip()
        if not name:
            return {"error": "invalid", "detail": "name required"}
        view_id = f"sv_{uuid.uuid4().hex[:12]}"
        created_at = datetime.now().isoformat()
        blob = json.dumps(filter_ or {}, ensure_ascii=False)
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO saved_views VALUES (?, ?, ?, ?, ?)",
                    (view_id, namespace, name, blob, created_at))
        except sqlite3.IntegrityError:
            # UNIQUE(namespace, name) 위반 — 같은 이름의 탐색이 이미 있다
            return {"error": "duplicate", "detail": name}
        logger.info(f"💾 Saved view created: {namespace}/{name}")
        return {"id": view_id, "namespace": namespace, "name": name,
                "filter": filter_ or {}, "created_at": created_at}

    def delete(self, namespace: str, view_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM saved_views WHERE namespace = ? AND id = ?",
                (namespace, view_id))
            return cur.rowcount > 0

    def delete_namespace(self, namespace: str) -> int:
        """네임스페이스 삭제 시 그 뷰를 모두 정리 — 유령 뷰 방지."""
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM saved_views WHERE namespace = ?", (namespace,))
            return cur.rowcount
