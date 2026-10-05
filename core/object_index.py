"""ES 객체 인덱스 — PG(진실)의 노드를 **검색·파셋용으로 투영**한다(축 5, P3).

노드 검색은 벡터가 아니라 BM25/키워드가 주다: 고유명사·조문번호처럼 문자 그대로
맞아야 하는 질의를 임베딩은 놓친다(ontology/CLAUDE.md 검색 스택 주석). 청크 벡터
검색(semantic_index/es_backend)과는 **별도 인덱스** `ontology-obj-{namespace}`:
  · type/trust = keyword  → terms aggregation(파셋)·정확 필터
  · name       = text(BM25) + name.kw(keyword, 정확/정렬)
  · aliases/definition = text(BM25 보조)

PG=진실, 이 인덱스는 파생 투영 — 언제든 sync 로 재생성한다. ES 없으면 degrade
(available()=False → 호출측이 PG substring 으로 폴백). 커널 경계: elasticsearch 는
lazy import(es extra).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from loguru import logger

OBJ_PREFIX = os.environ.get("ONTOLOGY_ES_OBJ_PREFIX", "ontology-obj")
DEFAULT_ES_URL = os.environ.get("ONTOLOGY_ES_URL", "http://localhost:9200")

_TEXT_PROPS = ("source", "definition")  # properties(jsonb)에서 끌어올 텍스트


def build_object_mapping() -> Dict[str, Any]:
    """인덱스 매핑 — **순수 함수**(살아있는 ES 불필요, 테스트 가능)."""
    return {
        "mappings": {
            "properties": {
                "namespace": {"type": "keyword"},
                "node_id": {"type": "keyword"},
                "type": {"type": "keyword"},
                "trust": {"type": "keyword"},
                "name": {"type": "text",
                         "fields": {"kw": {"type": "keyword"}}},
                "aliases": {"type": "text"},
                "definition": {"type": "text"},
            }
        }
    }


def build_search_body(q: Optional[str], node_type: Optional[str],
                      trust: Optional[str]) -> Dict[str, Any]:
    """검색 query 절 — **순수 함수**. q 있으면 BM25 멀티매치(name 가중),
    없으면 match_all. type/trust 는 정확 필터(filter 절, 점수 무관)."""
    must: List[Dict[str, Any]] = []
    ql = (q or "").strip()
    if ql:
        # 부분일치("이름에 그 글자가 들어감")를 확실히 잡고, fuzzy 는 쓰지 않는다.
        # fuzzy(AUTO)는 "삼층석탑→오층석탑"처럼 한 글자 다른 걸 끌어와 오히려
        # 노이즈였다. wildcard(name.kw) 가 substring 을 결정적으로 매칭하고,
        # match(name/aliases/definition)는 토큰 관련도로 랭킹을 보탠다.
        must.append({"bool": {"minimum_should_match": 1, "should": [
            {"wildcard": {"name.kw": {"value": f"*{ql}*",
                                      "case_insensitive": True, "boost": 6}}},
            {"match_phrase": {"name": {"query": ql, "boost": 3}}},
            {"match": {"name": {"query": ql, "boost": 2}}},
            {"match": {"aliases": {"query": ql}}},
            {"match": {"definition": {"query": ql}}},
        ]}})
    else:
        must.append({"match_all": {}})
    filt: List[Dict[str, Any]] = []
    if node_type:
        filt.append({"term": {"type": node_type}})
    if trust:
        filt.append({"term": {"trust": trust}})
    return {"bool": {"must": must, "filter": filt}}


_AGGS = {
    "by_type": {"terms": {"field": "type", "size": 200}},
    "by_trust": {"terms": {"field": "trust", "size": 10}},
}


def graph_to_docs(namespace: str, graph) -> List[Dict[str, Any]]:
    """NetworkX 그래프 → 색인 문서. PostgresGraphStore 의 컬럼/properties 분리와
    같은 규약(name/type/trust + properties 에서 aliases/definition)."""
    docs = []
    for nid, attrs in graph.nodes(data=True):
        aliases = attrs.get("aliases") or []
        docs.append({
            "namespace": namespace,
            "node_id": nid,
            "type": attrs.get("type", "") or "unknown",
            "trust": attrs.get("trust") or "unset",
            "name": attrs.get("name") or nid,
            "aliases": " ".join(str(a) for a in aliases),
            "definition": str(attrs.get("definition", "") or ""),
        })
    return docs


class ObjectIndex:
    """네임스페이스별 객체 인덱스. 인덱스명 `ontology-obj-{namespace}`."""

    def __init__(self, namespace: str, url: Optional[str] = None):
        self.namespace = namespace
        # ES 인덱스명은 소문자만 허용한다("AI-Coach" 같은 대문자 네임스페이스는
        # 그대로 쓰면 400 invalid_index_name → 노드 투영이 조용히 죽는다).
        # 인덱스명만 소문자로 파생하고 네임스페이스 정체성(graph/chunk_store)은
        # 원형 유지한다.
        self.index = f"{OBJ_PREFIX}-{namespace}".lower()
        self._url = url or DEFAULT_ES_URL
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from elasticsearch import Elasticsearch
                self._client = Elasticsearch(self._url, request_timeout=10)
            except Exception as e:
                logger.warning(f"⚠️ Elasticsearch 불가 ({e}) — 객체 인덱스 비활성")
                return None
        return self._client

    def available(self) -> bool:
        c = self.client
        if c is None:
            return False
        try:
            return bool(c.ping())
        except Exception as e:
            logger.warning(f"⚠️ Elasticsearch ping 실패 ({e})")
            return False

    def sync(self, docs: List[Dict[str, Any]]) -> int:
        """네임스페이스 인덱스를 문서로 **충실히 재생성**(delete index + bulk).
        per-namespace 인덱스라 통째 재생성이 가장 단순한 미러다. 성공 시 색인 수."""
        c = self.client
        if c is None:
            raise RuntimeError("elasticsearch unavailable")
        from elasticsearch import helpers
        if c.indices.exists(index=self.index):
            c.indices.delete(index=self.index)
        c.indices.create(index=self.index, **build_object_mapping())
        if docs:
            actions = [{"_index": self.index, "_id": d["node_id"], "_source": d}
                       for d in docs]
            helpers.bulk(c, actions)
            c.indices.refresh(index=self.index)
        logger.info(f"🔎 ES 객체 인덱스 동기화: {self.index} — {len(docs)} 문서")
        return len(docs)

    def delete(self) -> None:
        c = self.client
        if c is not None and c.indices.exists(index=self.index):
            c.indices.delete(index=self.index)

    def search(self, q: Optional[str] = None, node_type: Optional[str] = None,
               trust: Optional[str] = None, top_k: int = 50,
               offset: int = 0) -> Dict[str, Any]:
        """BM25 검색 + 타입/신뢰 파셋(aggregation). 반환:
        {total, items[{node_id,name,type,trust,score}], facets{type,trust}}."""
        c = self.client
        if c is None:
            raise RuntimeError("elasticsearch unavailable")
        res = c.search(
            index=self.index,
            query=build_search_body(q, node_type, trust),
            aggs=_AGGS, from_=offset, size=top_k, track_total_hits=True)
        hits = res.get("hits", {})
        items = []
        for h in hits.get("hits", []):
            src = h.get("_source", {})
            items.append({
                "node_id": src.get("node_id"),
                "name": src.get("name"),
                "type": src.get("type", ""),
                "trust": src.get("trust", "unset"),
                "score": h.get("_score")})
        aggs = res.get("aggregations", {})

        def _facet(key):
            return {b["key"]: b["doc_count"]
                    for b in aggs.get(key, {}).get("buckets", [])}

        total = hits.get("total", {})
        return {
            "namespace": self.namespace,
            "total": total.get("value", 0) if isinstance(total, dict) else total,
            "items": items,
            "facets": {"type": _facet("by_type"), "trust": _facet("by_trust")},
            "offset": offset, "limit": top_k,
        }


def project_graph(namespace: str, graph) -> Optional[int]:
    """편의: 그래프를 객체 인덱스로 투영(best-effort). ES 불가면 None."""
    idx = ObjectIndex(namespace)
    if not idx.available():
        return None
    return idx.sync(graph_to_docs(namespace, graph))
