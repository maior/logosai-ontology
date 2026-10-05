"""PostgreSQL 연결 — 온톨로지 인스턴스(노드·엣지)의 **진실 원본**.

축 5(저장 엔진 재설계)의 토대. 팔란티어가 불변 레이크를 진실로 두고 ES 를
투영으로 쓰는 것과 같은 원리로, 여기서는 PostgreSQL 이 ACID 진실 원본이고
Elasticsearch(`es_backend.py`)는 검색·파셋용 파생 투영이다. 인메모리 NetworkX
(`engines/`)는 폴백 + 빌드/추론 캐시로 강등된다.

커널 경계(CLAUDE.md): psycopg 는 **lazy import**(postgres extra) — 이 모듈을
import 해도 psycopg 가 없어야 죽지 않는다. 접속 불가·미설치 → `available()` False,
호출측은 InMemoryGraphStore 로 degrade(ES 백엔드와 동일한 degrade 계약).

env:
    ONTOLOGY_PG_DSN     예: postgresql://user:pw@db.example.com:5432/dbname
    ONTOLOGY_PG_SCHEMA  기본 'ontology' (기존 DB 안의 전용 스키마)
"""
from __future__ import annotations

import os
import re
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from loguru import logger

DEFAULT_SCHEMA = os.environ.get("ONTOLOGY_PG_SCHEMA", "ontology")
_IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def get_dsn() -> Optional[str]:
    """접속 문자열. 미설정이면 None (→ available() False → degrade)."""
    return os.environ.get("ONTOLOGY_PG_DSN") or None


def get_schema() -> str:
    """스키마명. env 로 재정의 가능하되 식별자 형식만 허용(주입 방지)."""
    schema = os.environ.get("ONTOLOGY_PG_SCHEMA", DEFAULT_SCHEMA)
    if not _IDENT_RE.match(schema):
        raise ValueError(f"invalid ONTOLOGY_PG_SCHEMA: {schema!r} "
                         "(소문자/숫자/밑줄, 숫자로 시작 불가)")
    return schema


def build_ddl(schema: str = DEFAULT_SCHEMA) -> str:
    """스키마+테이블+인덱스 생성 DDL — **순수 함수**(살아있는 DB 불필요, 테스트 가능).

    멱등(CREATE ... IF NOT EXISTS)이라 반복 실행해도 안전하다.
    KG 엔진(networkx.MultiDiGraph) 모델을 충실히 옮긴다:
      · node = node_id(opaque) + attrs{type, name, trust, properties}
      · edge = (source, predicate, target) + weight + properties (MultiDiGraph
               → predicate 를 PK 에 포함해 같은 쌍의 복수 관계 허용)
    """
    if not _IDENT_RE.match(schema):
        raise ValueError(f"invalid schema identifier: {schema!r}")
    return f"""
CREATE SCHEMA IF NOT EXISTS {schema};

-- 인스턴스 층: 노드 --------------------------------------------------------
CREATE TABLE IF NOT EXISTS {schema}.node (
    namespace   text        NOT NULL,
    node_id     text        NOT NULL,
    type        text        NOT NULL DEFAULT 'unknown',
    name        text,
    trust       text        NOT NULL DEFAULT 'unset',
    properties  jsonb       NOT NULL DEFAULT '{{}}'::jsonb,
    content_hash text,       -- 증분 미러용 내용 해시 (변경 감지)
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (namespace, node_id)
);
-- 기존 테이블 마이그레이션(컬럼 없던 설치본) — 멱등
ALTER TABLE {schema}.node ADD COLUMN IF NOT EXISTS content_hash text;
CREATE INDEX IF NOT EXISTS node_ns_type   ON {schema}.node (namespace, type);
CREATE INDEX IF NOT EXISTS node_props_gin  ON {schema}.node USING gin (properties);

-- 인스턴스 층: 엣지 (방향 있는 다중 그래프) --------------------------------
CREATE TABLE IF NOT EXISTS {schema}.edge (
    namespace   text        NOT NULL,
    source_id   text        NOT NULL,
    predicate   text        NOT NULL,
    target_id   text        NOT NULL,
    weight      real        NOT NULL DEFAULT 1.0,
    properties  jsonb       NOT NULL DEFAULT '{{}}'::jsonb,
    content_hash text,       -- 증분 미러용 내용 해시
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (namespace, source_id, predicate, target_id)
);
ALTER TABLE {schema}.edge ADD COLUMN IF NOT EXISTS content_hash text;
-- 이웃 확장(out) · 역방향(in, is_a 조상) · 술어 폐포를 전부 인덱스로:
CREATE INDEX IF NOT EXISTS edge_out  ON {schema}.edge (namespace, source_id);
CREATE INDEX IF NOT EXISTS edge_in   ON {schema}.edge (namespace, target_id);
CREATE INDEX IF NOT EXISTS edge_pred ON {schema}.edge (namespace, predicate);
""".strip()


def _psycopg():
    """psycopg(v3) 를 lazy import. 없으면 None (degrade)."""
    try:
        import psycopg  # noqa: F401
        return psycopg
    except Exception as e:  # ImportError 포함
        logger.warning(f"⚠️ psycopg 미설치/불가 ({e}) — PostgreSQL 백엔드 비활성")
        return None


@contextmanager
def connect() -> Iterator[Any]:
    """psycopg 연결 컨텍스트. DSN 미설정/psycopg 부재/접속 실패 시 예외."""
    dsn = get_dsn()
    if not dsn:
        raise RuntimeError("ONTOLOGY_PG_DSN 미설정")
    pg = _psycopg()
    if pg is None:
        raise RuntimeError("psycopg 미설치 (postgres extra)")
    conn = pg.connect(dsn, connect_timeout=5)
    try:
        yield conn
    finally:
        conn.close()


def available() -> bool:
    """PG 백엔드를 쓸 수 있는가 — DSN + psycopg + 실제 접속까지 확인.

    ES `available()` 과 같은 계약: 조용히 True 를 주지 않는다(가짜 성공 금지).
    실패는 WARNING 만 남기고 False — 호출측이 memory 로 degrade 한다.
    """
    if not get_dsn() or _psycopg() is None:
        return False
    try:
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception as e:
        logger.warning(f"⚠️ PostgreSQL 접속 불가 ({e}) — memory 로 degrade")
        return False


def ensure_schema(schema: Optional[str] = None) -> bool:
    """스키마·테이블·인덱스를 멱등 생성. 성공 True, 접속 실패 등은 False."""
    schema = schema or get_schema()
    try:
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(build_ddl(schema))
            conn.commit()
        logger.info(f"✅ PostgreSQL 온톨로지 스키마 준비 완료: {schema}")
        return True
    except Exception as e:
        logger.error(f"❌ 스키마 생성 실패 ({e})")
        return False
