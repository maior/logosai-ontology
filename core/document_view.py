"""문서 뷰 — 문서를 **일급으로 조회·대조**하되, KG 노드로 만들지 않는다.

**왜 노드가 아닌가 (측정된 위험).** 허브가 순위를 오염시키는 것은 이 저장소에서
실측됐다 — PPR 원 질량 정렬에서 총칙 조문(허브 청크)이 정답을 밀어내 lift 교정이
필요했다(graph_propagation.lift). Document 를 노드로 만들어 그 문서의 청크
수백 개와 링크하면 **슈퍼허브**가 되어 채널 B(노드에 달린 청크)와 확산을 오염시킨다.

"일급"의 실질 — 주소·조회·필터·대조 — 은 노드 없이 성립한다. 문서는 이미
`chunk.source` 축으로 데이터에 존재하고, 빠져 있던 것은 **그걸 묻는 API** 였다:
  · 문서 목록 + 문서별 상태 (이 모듈)
  · 문서 간 대조 (이 모듈) — "제안요청서가 요구한 것 중 제안서가 빠뜨린 것"
  · 문서 필터 검색 (`chunk_index`·`graph_retrieval` 의 source 파라미터)

개체의 문서 귀속 = **그 문서의 청크에 근거 링크가 있는가.** 결정적이고 LLM 0콜이다.
근거 링크가 성기면(커버리지 낮음) 대조도 성기다 — 그 한계는 문서별 coverage 로
같이 보고한다(숨기지 않는다).

전부 순수 함수 (graph·chunks 를 인자로) — 커널 안이고, 실패는 빈 결과 (never raise).
"""

from typing import Any, Dict, Iterable, List

from loguru import logger


def document_views(graph, chunks: Iterable[Any]) -> List[Dict[str, Any]]:
    """청크를 source 로 묶은 문서별 상태.

    섹션은 **개수만** 센다 — 전체를 나열하면 문서 목록 응답이 원문 크기로 불어난다.
    그래프에서 지워진 노드를 가리키는 링크(dangling)는 세지 않는다 — 세면 nodes 가
    유령을 포함해 거짓이 된다.
    """
    try:
        node_ids = set(graph.nodes())
        docs: Dict[str, Dict[str, Any]] = {}
        for chunk in (chunks or []):
            source = str(getattr(chunk, "source", "") or "")
            if not source:
                continue
            doc = docs.setdefault(source, {
                "source": source, "chunks": 0, "linked_chunks": 0,
                "_nodes": set(), "_sections": set()})
            doc["chunks"] += 1
            linked = [n for n in (getattr(chunk, "node_ids", None) or [])
                      if n in node_ids]
            if linked:
                doc["linked_chunks"] += 1
                doc["_nodes"].update(linked)
            section = str(getattr(chunk, "section", "") or "").strip()
            if section:
                doc["_sections"].add(section)

        out: List[Dict[str, Any]] = []
        for source in sorted(docs):
            doc = docs[source]
            out.append({
                "source": source,
                "chunks": doc["chunks"],
                "linked_chunks": doc["linked_chunks"],
                "coverage": (doc["linked_chunks"] / doc["chunks"]
                             if doc["chunks"] else 0.0),
                "nodes": len(doc["_nodes"]),
                "sections": len(doc["_sections"]),
            })
        return out
    except Exception as e:
        logger.warning(f"⚠️ Document view failed ({e})")
        return []


def coverage_map(graph, chunks: Iterable[Any], source: str = "",
                 text_head: int = 90) -> Dict[str, Any]:
    """커버리지 지도 — 문서를 **원문 순서대로** 편 청크 스트립의 재료.

    document_views 가 문서당 요약 한 줄이라면, 이것은 그 안을 편다: 각 청크가
    문서의 어느 위치(char_start)에 있고, 근거 링크가 몇 개인지. "커버리지 70%"
    가 아니라 **"비어 있는 30%가 어느 절인가"** 를 묻는 화면의 데이터다 —
    커버리지 회복 루프(review/coverage)의 타깃 선정이 이 답에 걸려 있다.

    규칙:
      · 순서 = char_start (원문 축 — 축 2 의 오프셋 불변식이 여기서도 자다).
        오프셋이 없으면 적재 순서로 폴백하되 순서 신뢰 여부를 ordered 로 알린다.
      · links 는 dangling 제외 (document_views 와 같은 이유 — 유령은 거짓).
      · 본문은 머리(text_head)만 — 스트립 hover 용. 전문은 GET /chunks/{id}.
      · source 지정 시 그 문서만, 미지정이면 전 문서. 없는 source 는
        빈 documents 가 아니라 **소리내어** 알린다 (compare 와 같은 이유 —
        오타가 "청크 0 = 전부 커버됨"으로 오독된다).
    """
    try:
        node_ids = set(graph.nodes())
        chunk_list = list(chunks or [])
        wanted = (source or "").strip()
        if wanted:
            sources = {str(getattr(c, "source", "") or "") for c in chunk_list}
            if wanted not in sources:
                return {"error": "source_not_found",
                        "detail": f"'{wanted}' 는 이 네임스페이스에 없다 — "
                                  f"보유 문서: {sorted(s for s in sources if s)}"}

        docs: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in chunk_list:
            src = str(getattr(chunk, "source", "") or "")
            if not src or (wanted and src != wanted):
                continue
            start = getattr(chunk, "char_start", None)
            end = getattr(chunk, "char_end", None)
            text = str(getattr(chunk, "text", "") or "")
            links = [n for n in (getattr(chunk, "node_ids", None) or [])
                     if n in node_ids]
            docs.setdefault(src, []).append({
                "chunk_id": getattr(chunk, "chunk_id", ""),
                "section": str(getattr(chunk, "section", "") or "").strip(),
                "char_start": start if isinstance(start, int) else None,
                "char_end": end if isinstance(end, int) else None,
                "links": len(links),
                "node_ids": sorted(links),
                "text_head": text[:text_head],
            })

        out = []
        for src in sorted(docs):
            rows = docs[src]
            ordered = all(r["char_start"] is not None for r in rows)
            if ordered:
                rows.sort(key=lambda r: r["char_start"])
            for i, r in enumerate(rows, 1):
                r["order"] = i
            linked = sum(1 for r in rows if r["links"] > 0)
            out.append({
                "source": src,
                "chunks": rows,
                "total": len(rows),
                "linked": linked,
                "coverage": linked / len(rows) if rows else 0.0,
                # 순서를 신뢰할 수 없으면 화면이 알아야 한다 — "문서 순서"라는
                # 약속이 거짓인 채 스트립을 그리면 빈 구간 판독이 엉뚱해진다.
                "ordered": ordered,
            })
        return {"documents": out}
    except Exception as e:
        logger.warning(f"⚠️ Coverage map failed ({e})")
        return {"documents": []}


def _canonical_groups(graph, node_ids: Iterable[str]) -> Dict[str, List[str]]:
    """대조용 개체 정준화 — 대표 노드 → 구성원 목록 (union-find).

    **실측이 요구했다.** PROJ-A 요구서↔제안서의 identity 공통은 6인데 표기
    변형("통합저장소"↔"통합 저장소")을 묶으면 12 — 공통의 절반이 "같은 개념,
    다른 표기"로 차이에 잘못 계상되고 있었다. LLM 추출은 문서마다 띄어쓰기가
    흔들린다(비결정성의 알려진 모양).

    묶는 규칙 둘 — 둘 다 결정적이다:
      · **(타입, 정규화명) 동일** — `graph_health.normalize_name` 재사용
        (규칙을 두 벌 두면 '같다'의 정의가 갈라진다). 타입이 다르면 묶지 않는다 —
        타입 무시 후보 발굴은 graph_health 중복 검사의 몫이고, 자동 묶음은
        보수적이어야 한다. 포함 관계("갑상선암" ⊂ "중증 갑상선암")도 묶지 않는다.
      · **sameAs 엣지** — 사람이 검수해 넣은 동일시 선언.

    묶음은 본질적으로 전이적이다(A=B, B=C 면 한 무리). 질의 확장의 sameAs 는
    전이를 금지하지만(오염 위험), 여기는 **보고서**다 — members 로 묶음을
    드러내므로 잘못된 묶음이 눈에 보이고, 판단은 사람이 한다.
    """
    from .graph_health import normalize_name
    from .graph_retrieval import SYNONYM_PREDICATE

    ids = sorted(set(node_ids))
    parent = {nid: nid for nid in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            # 사전순 작은 쪽이 대표 — 결정론성 (실행마다 대표가 바뀌면
            # 대조 결과를 비교할 수 없다)
            if ry < rx:
                rx, ry = ry, rx
            parent[ry] = rx

    by_key: Dict[tuple, str] = {}
    for nid in ids:
        attrs = graph.nodes[nid]
        # normalize_name 은 node_id 를 받아 첫 ":" 앞을 타입으로 떼어낸다 —
        # 이름을 넘길 때는 가짜 접두사를 붙여 이름 속 ":" 가 잘리지 않게 한다.
        key = (str(attrs.get("type") or ""),
               normalize_name(f"_:{attrs.get('name') or nid.split(':', 1)[-1]}"))
        if key[1]:
            if key in by_key:
                union(by_key[key], nid)
            else:
                by_key[key] = nid

    id_set = set(ids)
    for source, target, attrs in graph.edges(data=True):
        if attrs.get("predicate") == SYNONYM_PREDICATE \
                and source in id_set and target in id_set:
            union(source, target)

    groups: Dict[str, List[str]] = {}
    for nid in ids:
        groups.setdefault(find(nid), []).append(nid)
    return {rep: sorted(members) for rep, members in groups.items()}


def compare_documents(graph, chunks: Iterable[Any],
                      source_a: str, source_b: str) -> Dict[str, Any]:
    """두 문서의 개체 대조 — A에만 근거가 있는 개체 / B에만 / 공통.

    **정준화 위에서 대조한다** (`_canonical_groups`) — 같은 개념이 문서마다 다른
    표기로 추출되면 identity 대조는 그걸 전부 "차이"로 오보고한다. 묶인 항목은
    `members` 로 드러낸다 — 조용히 합치면 잘못된 묶음을 볼 수 없다.

    **없는 문서·같은 문서는 소리내어 거부한다.** 오타 파일명이 빈 결과를 내면
    "전부 커버됨"으로 오독된다 — evaluate 의 0건 정직 보고와 같은 부류.

    어느 문서에도 근거가 없는 노드(고아)는 대조 대상이 아니다 — 섞으면
    "빠뜨렸다"가 "원래 어디에도 없었다"와 구별되지 않는다.

    각 개체에 근거 위치(청크·섹션)를 실어 준다 — 검수자가 원문을 확인할 수
    있어야 대조 결과가 판단의 재료가 된다.
    """
    source_a = (source_a or "").strip()
    source_b = (source_b or "").strip()
    if source_a == source_b:
        return {"error": "invalid", "detail": "같은 문서끼리는 대조할 수 없다"}
    try:
        chunk_list = list(chunks or [])
        sources = {str(getattr(c, "source", "") or "") for c in chunk_list}
        for wanted in (source_a, source_b):
            if wanted not in sources:
                return {"error": "source_not_found",
                        "detail": f"'{wanted}' 는 이 네임스페이스에 없다 — "
                                  f"보유 문서: {sorted(s for s in sources if s)}"}

        node_ids = set(graph.nodes())
        # 노드 → 문서별 근거 [(chunk_id, section)]
        evidence: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
        for chunk in chunk_list:
            source = str(getattr(chunk, "source", "") or "")
            if source not in (source_a, source_b):
                continue
            for nid in (getattr(chunk, "node_ids", None) or []):
                if nid not in node_ids:
                    continue          # dangling — 유령을 대조에 넣지 않는다
                evidence.setdefault(nid, {source_a: [], source_b: []})
                evidence[nid][source].append({
                    "chunk_id": getattr(chunk, "chunk_id", ""),
                    "section": str(getattr(chunk, "section", "") or "")})

        groups = _canonical_groups(graph, evidence.keys())

        def _entry(rep: str, members: List[str],
                   key_a: bool, key_b: bool) -> Dict[str, Any]:
            attrs = graph.nodes[rep]
            row: Dict[str, Any] = {
                "node_id": rep,
                "name": attrs.get("name", rep.split(":", 1)[-1]),
                "type": attrs.get("type", "")}
            others = [m for m in members if m != rep]
            if others:
                # 묶임을 드러낸다 — 조용히 합치면 잘못된 묶음을 볼 수 없다.
                row["members"] = others
            if key_a:
                row["evidence_a"] = [ev for m in members
                                     for ev in evidence[m][source_a]]
            if key_b:
                row["evidence_b"] = [ev for m in members
                                     for ev in evidence[m][source_b]]
            return row

        only_a, only_b, shared = [], [], []
        for rep in sorted(groups):
            members = groups[rep]
            in_a = any(evidence[m][source_a] for m in members)
            in_b = any(evidence[m][source_b] for m in members)
            if in_a and in_b:
                shared.append(_entry(rep, members, True, True))
            elif in_a:
                only_a.append(_entry(rep, members, True, False))
            elif in_b:
                only_b.append(_entry(rep, members, False, True))

        return {"source_a": source_a, "source_b": source_b,
                "only_a": only_a, "only_a_total": len(only_a),
                "only_b": only_b, "only_b_total": len(only_b),
                "shared": shared, "shared_total": len(shared)}
    except Exception as e:
        logger.warning(f"⚠️ Document compare failed ({e})")
        return {"error": "internal", "detail": str(e)[:200]}
