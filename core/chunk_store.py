"""
Chunk store — 원문 보존 + span provenance.

축 2. 빌더는 추출이 끝나면 원문을 버렸다. 노드 attrs 와 source 경로,
chunk_index 만 남았고 그 결과:
- 검색 결과로 보여줄 문장이 없다 → "비정형 데이터를 찾는다"가 성립하지 않는다.
- 인용이 불가능하다 → aicoach 가 rag/legal.py 를 따로 만든 이유가 이것이다
  (인용이 p.47 이 아니라 제21조를 가리켜야 한다).
- 데이터셋 추출이 그래프를 되읽는 수준에 머문다 → 근거 문장이 없다.

여기서 노드 → 청크 → 원문 문자 오프셋까지 내려가는 사슬을 만든다.

설계 선택과 근거:
- **chunk_id 는 내용 해시** — 같은 문서를 다시 빌드해도 같은 id 가 나와
  중복이 쌓이지 않는다 (aicoach rag/ingest.py 의 sha1 _id 규약 이식).
  재빌드에서 다른 개체가 추출되면 node_ids 는 합집합으로 늘어난다.
- **JSONL** — 한 줄이 깨져도 나머지는 살아남는다. KG 는 통짜 JSON 이라
  파일 하나가 깨지면 그래프 전체를 잃는데, 원문은 재추출 비용이 훨씬 크므로
  같은 실패 방식을 물려받지 않는다. 쓰기는 KG 와 같은 원자적 교체(.tmp → rename).
- **네임스페이스별 싱글턴** — KG 엔진과 같은 수명·같은 경계.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger

_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"


def chunk_id_for(source: str, index: int, text: str) -> str:
    """내용 기반 결정적 id. 같은 (출처, 순번, 본문머리) → 같은 id.

    본문 전체가 아니라 앞 40자만 넣는 것은 aicoach 규약 그대로다 — 긴 청크
    해싱 비용을 피하면서 (source, index) 충돌만 걸러내면 충분하다.
    """
    raw = f"{source}::{index}::{text[:40]}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class StoredChunk:
    """저장된 청크 — 원문 + provenance + 이 청크에서 추출된 노드들."""
    chunk_id: str
    text: str
    source: str = ""
    index: int = 0
    section: str = ""
    char_start: int = 0
    char_end: int = 0
    node_ids: List[str] = field(default_factory=list)
    # 출처의 신뢰 등급 (authoritative/summary/…). 검색·인용 시 요약서 청크를
    # 원본 청크보다 앞세우지 않기 위한 provenance (aicoach layer 실증).
    trust: str = ""
    # 도메인/출처 메타 — layer(L1규제~L4교육)·tenant 등 임의 provenance.
    # 일반 dict 로 두어 커널을 특정 도메인에 묶지 않는다(aicoach layer 이식).
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredChunk":
        return cls(
            chunk_id=data["chunk_id"],
            text=data.get("text", ""),
            source=data.get("source", ""),
            index=int(data.get("index", 0)),
            section=data.get("section", ""),
            char_start=int(data.get("char_start", 0)),
            char_end=int(data.get("char_end", 0)),
            node_ids=list(data.get("node_ids") or []),
            trust=data.get("trust", ""),  # 축 2 시절 파일에는 없다 — 기본 ""
            meta=dict(data.get("meta") or {}),  # 이전 파일에는 없다 — 기본 {}
        )


class ChunkStore:
    """네임스페이스 하나의 원문 청크 저장소."""

    def __init__(self, namespace: str = "default", path=None):
        self.namespace = namespace
        self._path = Path(path) if path else None
        self._chunks: Dict[str, StoredChunk] = {}
        self._by_node: Dict[str, List[str]] = {}  # node_id → [chunk_id]

    # ─── 경로 ────────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        """네임스페이스별 저장 파일. KG 체크포인트와 같은 디렉터리에 둔다."""
        if self._path is not None:
            return self._path
        return _DEFAULT_DATA_DIR / f"chunks_{self.namespace}.jsonl"

    # ─── 쓰기 ────────────────────────────────────────────────────────

    def add(self, chunk, node_ids: Iterable[str] = (), trust: str = "",
            meta: Optional[Dict[str, Any]] = None) -> str:
        """청크를 저장하고 chunk_id 를 돌려준다. 같은 청크면 멱등.

        재빌드에서 다른 개체가 추출되는 경우가 있으므로 node_ids 는
        덮어쓰지 않고 합집합으로 병합한다. trust 는 비어 있을 때만 채운다 —
        청크는 출처가 하나이므로 등급이 바뀌는 것은 재감식뿐이다.
        meta(layer 등)는 넘어온 키만 병합한다 — chunk_id 를 안 바꾸므로
        기존 벡터 캐시를 무효화하지 않고 provenance 를 보강할 수 있다.
        """
        cid = chunk_id_for(chunk.source, chunk.index, chunk.text)
        existing = self._chunks.get(cid)
        eff_meta = dict(getattr(chunk, "meta", None) or {})
        if meta:
            eff_meta.update(meta)

        if existing is None:
            existing = StoredChunk(
                chunk_id=cid, text=chunk.text, source=chunk.source,
                index=chunk.index, section=getattr(chunk, "section", ""),
                char_start=getattr(chunk, "char_start", 0),
                char_end=getattr(chunk, "char_end", 0),
                trust=trust, meta=eff_meta,
            )
            self._chunks[cid] = existing
        else:
            if trust and not existing.trust:
                existing.trust = trust
            if eff_meta:
                existing.meta.update(eff_meta)

        for node_id in node_ids:
            if node_id not in existing.node_ids:
                existing.node_ids.append(node_id)
            bucket = self._by_node.setdefault(node_id, [])
            if cid not in bucket:
                bucket.append(cid)
        return cid

    def link_node(self, chunk_id: str, node_id: str) -> bool:
        """이 청크가 이 노드의 근거임을 사후에 기록한다 (커버리지 gap 회복).

        add() 도 node_ids 를 합집합으로 병합하지만, 그 경로는 **저장**이다 —
        trust/meta 규칙을 들고 있고, StoredChunk 를 넘기는 것은 그것이 Chunk 와
        같은 속성 이름을 가진 우연에 의존한다. 링크만 필요한 호출자에게 저장
        경로를 쓰게 하면 나중에 add() 의 규칙이 바뀔 때 조용히 끌려간다.

        없는 청크는 False — 없는 청크에 링크를 만들면 청크 쪽이 아니라 노드 쪽
        참조만 남아 근거를 클릭했을 때 빈 화면이 된다. 이미 이어져 있으면
        False(멱등): True 는 "이번에 새로 이었다"는 뜻이라야 호출자가 몇 건을
        복구했는지 셀 수 있다.
        """
        chunk_id = (chunk_id or "").strip()
        node_id = (node_id or "").strip()
        if not chunk_id or not node_id:
            return False
        stored = self._chunks.get(chunk_id)
        if stored is None:
            return False
        if node_id in stored.node_ids:
            return False
        stored.node_ids.append(node_id)
        bucket = self._by_node.setdefault(node_id, [])
        if chunk_id not in bucket:
            bucket.append(chunk_id)
        return True

    def relabel_node(self, old_node_id: str, new_node_id: str) -> List[str]:
        """노드 참조를 갈아끼운다 — 노드 병합의 청크 쪽 절반.

        그래프에서 노드를 지우고 이걸 안 하면 청크의 node_ids 가 허공을 가리켜
        (graph_health 의 dangling_node_refs) 인용을 클릭하면 빈 화면이 된다.

        **역색인(_by_node)까지 갱신해야 한다** — node_ids 만 제자리 수정하면
        chunks_for_node(지워진 노드)가 계속 근거를 돌려준다. chunk_id 는 내용
        해시라 바뀌지 않으므로 벡터 캐시는 무효화되지 않는다.

        돌려주는 것은 손댄 chunk_id 들 — 0건이면 그 노드는 근거가 없었다는 뜻이다.
        """
        old_node_id = (old_node_id or "").strip()
        new_node_id = (new_node_id or "").strip()
        if not old_node_id or not new_node_id or old_node_id == new_node_id:
            return []
        touched: List[str] = []
        for cid in list(self._by_node.get(old_node_id, [])):
            stored = self._chunks.get(cid)
            if stored is None:
                continue
            rewritten: List[str] = []
            for ref in stored.node_ids:        # 순서 보존 dedup — 둘 다 가리키던
                candidate = new_node_id if ref == old_node_id else ref
                if candidate not in rewritten:  # 청크는 병합 후 같은 id 가 둘 된다
                    rewritten.append(candidate)
            stored.node_ids = rewritten
            bucket = self._by_node.setdefault(new_node_id, [])
            if cid not in bucket:
                bucket.append(cid)
            touched.append(cid)
        self._by_node.pop(old_node_id, None)
        return touched

    def delete_by_source(self, source: str) -> int:
        """한 소스(문서)의 청크를 전부 지운다 — 소스별 멱등 재적재(Phase 1-3).

        청크는 source 와 1:1(내용 해시 id)이라 소스 단위 삭제가 안전하다. 같은
        문서를 다시 올릴 때(내용이 바뀌면 새 id 가 생겨) 옛 청크가 쌓이는 것을
        막는다 — 재적재 전에 호출해 '교체' 의미를 준다. _by_node 역인덱스도 정리."""
        if not source:
            return 0
        victims = [cid for cid, c in self._chunks.items() if c.source == source]
        vset = set(victims)
        for cid in victims:
            del self._chunks[cid]
        if vset:
            for node_id in list(self._by_node):
                kept = [c for c in self._by_node[node_id] if c not in vset]
                if kept:
                    self._by_node[node_id] = kept
                else:
                    del self._by_node[node_id]
        return len(victims)

    def clear(self) -> None:
        """저장소를 비운다 (rebuild 경로 — KG.clear() 와 짝)."""
        self._chunks.clear()
        self._by_node.clear()

    # ─── 읽기 ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._chunks)

    def get(self, chunk_id: str) -> Optional[StoredChunk]:
        return self._chunks.get(chunk_id)

    def all(self) -> List[StoredChunk]:
        return list(self._chunks.values())

    def chunks_for_node(self, node_id: str) -> List[StoredChunk]:
        """이 노드가 추출된 원문 청크들 — 인용·근거의 진입점."""
        return [self._chunks[cid] for cid in self._by_node.get(node_id, [])
                if cid in self._chunks]

    def nodes_for_chunk(self, chunk_id: str) -> List[str]:
        stored = self._chunks.get(chunk_id)
        return list(stored.node_ids) if stored else []

    def search_text(self, query: str, top_k: int = 5) -> List[StoredChunk]:
        """부분문자열 폴백 검색.

        임베딩·ES 가 없어도 원문이 있으면 최소한의 구절 검색은 되게 한다.
        축 3 의 ES 백엔드가 이 자리를 대체하며, 여기는 의존성 0 인 바닥선이다.
        """
        if not query or not query.strip():
            return []
        needle = query.strip().lower()
        hits = [c for c in self._chunks.values() if needle in c.text.lower()]
        return hits[:top_k]

    # ─── 영속성 ──────────────────────────────────────────────────────

    def save_to_disk(self, path=None) -> bool:
        """JSONL 로 저장. KG 와 같은 원자적 교체(.tmp → rename)."""
        try:
            save_path = Path(path) if path else self.path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = save_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                for stored in self._chunks.values():
                    f.write(json.dumps(asdict(stored), ensure_ascii=False) + "\n")
            tmp_path.replace(save_path)
            logger.info(f"💾 Chunk store saved: {len(self._chunks)} chunks "
                        f"→ {save_path}")
            return True
        except Exception as e:
            logger.error(f"Chunk store save failed: {e}")
            return False

    def load_from_disk(self, path=None) -> bool:
        """JSONL 로드. 깨진 줄은 건너뛴다 — 한 줄 때문에 원문 전체를 잃지 않는다."""
        load_path = Path(path) if path else self.path
        if not load_path.exists():
            logger.info(f"No chunk store at {load_path} — starting fresh")
            return False
        try:
            self.clear()
            skipped = 0
            with open(load_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        stored = StoredChunk.from_dict(json.loads(line))
                    except Exception:
                        skipped += 1
                        continue
                    self._chunks[stored.chunk_id] = stored
                    for node_id in stored.node_ids:
                        self._by_node.setdefault(node_id, []).append(stored.chunk_id)
            if skipped:
                logger.warning(f"⚠️ Chunk store: skipped {skipped} corrupt line(s)")
            logger.info(f"📂 Chunk store loaded: {len(self._chunks)} chunks "
                        f"← {load_path}")
            return True
        except Exception as e:
            logger.error(f"Chunk store load failed: {e}")
            return False


# ─── 네임스페이스 싱글턴 ────────────────────────────────────────────

_stores: Dict[str, ChunkStore] = {}


def get_chunk_store(namespace: str = "default") -> ChunkStore:
    """네임스페이스별 공유 ChunkStore. 첫 호출 시 디스크에서 로드한다
    (KG 엔진 get_knowledge_graph_engine 과 같은 수명·같은 규약)."""
    if namespace not in _stores:
        store = ChunkStore(namespace=namespace)
        store.load_from_disk()
        _stores[namespace] = store
        logger.info(f"📚 Chunk store initialized (namespace={namespace})")
    return _stores[namespace]


def reset_chunk_stores() -> None:
    """싱글턴 초기화 — 테스트 격리용."""
    _stores.clear()
