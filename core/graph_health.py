"""그래프 건강 진단 — 근거 공백과 중복 노드를 **제시**한다 (판정하지 않는다).

축 2 가 "어디서 알았는가"(청크 provenance)를, 축 4 가 "온톨로지로 조건화된
검색"을 만들었다. 그런데 그 둘을 잇는 **노드↔청크 링크가 성기면** 그래프 채널은
코퍼스의 일부만 보고 돈다. 실측(ins_cancer_demo): 청크 92 중 27(29%)만 노드를
냈고, 노드 106 중 34(32%)는 근거 청크가 없었다. 지표는 멀쩡해 보였다 — 아무도
이 비율을 재지 않았기 때문이다.

세 가지 신호를 낸다:
  · orphan_nodes     — 그래프엔 있는데 근거 청크가 없는 노드 (인용 불가)
  · unlinked_chunks  — 원문은 있는데 아무 노드도 못 낸 청크 (그래프 채널 사각)
  · duplicate_clusters — 같은 것의 다른 이름 후보 (골든셋·인용을 조용히 깨뜨림)

**합치지 않는다.** 자동 병합은 되돌릴 수 없고, 잘못 합치면 인용이 엉뚱한 조항을
가리킨다 — 검수 묘비(review_store)와 같은 원칙으로 사람이 정한다.

**하드코딩 금지**: 정규화는 구조(타입 접두사·괄호·공백·문장부호)만 본다. 도메인
어휘나 코드 체계(ICD 등) 패턴을 넣지 않는다 — 넣는 순간 보험 문서 전용 커널이
된다. 그래서 의미는 같지만 표기가 전혀 다른 쌍(`유방암` ↔ `유방의 악성 신생물`)은
**여기서 못 잡는다**. 그건 임베더가 필요한 별개 층이고, 못 잡는다고 적어 둔다.

**consistency_checker 와의 관계 (세 번째 구현을 막기 위한 기록)**:
그쪽 `name_type_conflict` 는 같은 이름이 다른 타입에 있는 경우를 이미 잡는다.
정규화가 공백 압축 + casefold 뿐이라 `Disease:장해` ↔ `InsuranceTerm:장해` 는
겹치지만, 괄호 표기가 다른 쌍(`C50( 유방의 악성 신생물 )` ↔ `유방의 악성 신생물`)과
이름 계열(`납입최고` 6형제)은 놓친다 — 실제로 골든셋을 깨뜨린 것이 바로 그 쌍이다.
역할 분담: **consistency_checker = 검수자의 작업 큐**(건별 severity), **여기 =
관리자의 건강 요약**(비율 + 표본). 겹치는 신호는 cross-type 한 종뿐이고, 중복 노드를
검수 큐에도 올리려면 이 모듈의 identity_keys 를 그쪽에서 재사용하라 — 정규화
로직을 다시 쓰지 말 것.
"""

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

# 괄호 안은 문서에서 '같은 것의 다른 표기'를 적는 자리다(`C50( 유방의 악성 신생물 )`).
# 한글/전각 괄호까지 포함 — 한국어 문서에서 실제로 섞여 나온다.
_PARENS = re.compile(r"[（(\[【]([^)）\]】]*)[)）\]】]")
_NONWORD = re.compile(r"[\W_]+", re.UNICODE)

# 괄호 조각의 최소 길이. 1글자 조각("A", "가")으로 묶으면 무관한 노드가 무더기로
# 붙는다 — 과합침이 미합침보다 위험하다(사람이 되돌려야 하므로).
MIN_FRAGMENT_LEN = 2

# 한 괄호 조각이 이만큼 넘는 노드를 잇는다면 그건 신원이 아니라 **주석**이다
# ("(제1항)" 같은). 측정값이 아니라 과합침 방어용 상한이므로 호출부가 덮을 수 있다.
DEFAULT_MAX_FRAGMENT_FANOUT = 4

# '추출 실패'로 셀 최소 본문 길이. 머리말·페이지번호까지 실패로 세면 커버리지가
# 거짓말을 한다. 관찰에서 고른 출발점이지 측정값이 아니다.
DEFAULT_MIN_CHUNK_LEN = 200


def _squash(text: Optional[str]) -> str:
    """공백·문장부호·대소문자를 지운 비교용 형태. 접두사는 건드리지 않는다."""
    if not isinstance(text, str):
        return ""
    return _NONWORD.sub("", text).casefold()


def _strip_type(node_id: str) -> str:
    """`Disease:유방암` → `유방암`. 접두사가 없으면 그대로 (통째로 날리지 않는다)."""
    return node_id.split(":", 1)[1] if ":" in node_id else node_id


def normalize_name(node_id: Optional[str]) -> str:
    """노드 id 의 비교용 이름 — 타입 접두사를 떼고 표기 흔들림을 지운다.

    타입을 떼는 이유: `InsuranceTerm:장해` 와 `Disease:장해` 는 타입이 갈렸을 뿐
    같은 말일 수 있다. 어느 쪽이 옳은지는 사람이 정하지만, **후보로는 올라와야**
    한다 — 타입까지 같아야 묶으면 이 사례가 영영 안 보인다.
    """
    if not isinstance(node_id, str) or not node_id.strip():
        return ""
    return _squash(_strip_type(node_id))


def identity_keys(node_id: Optional[str]) -> Set[str]:
    """이 노드가 '같은 것'으로 만날 수 있는 이름들.

    전체 이름 + 괄호 밖 + 괄호 안. 괄호 안을 버리면 `C50( 유방의 악성 신생물 )`
    은 코드만 남아 `유방의 악성 신생물` 과 영영 못 만난다 — 실제로 골든셋 정답이
    근거를 못 찾은 원인이 이것이었다.
    """
    full = normalize_name(node_id)
    if not full:
        return set()
    keys = {full}
    name = _strip_type(node_id)
    inside = _PARENS.findall(name)
    outside = _PARENS.sub(" ", name)
    for fragment in (outside, *inside):
        key = _squash(fragment)
        if len(key) >= MIN_FRAGMENT_LEN:
            keys.add(key)
    return keys


def duplicate_clusters(
        node_ids: Iterable[str],
        max_fragment_fanout: int = DEFAULT_MAX_FRAGMENT_FANOUT,
) -> List[List[str]]:
    """같은 개념일 수 있는 노드 묶음 — 사람이 볼 **후보** 목록.

    이름을 공유하면 잇고(union-find), 전이적으로 한 군집으로 만든다: A~B, B~C 가
    두 군집으로 쪼개지면 사람이 같은 판단을 두 번 해야 한다.

    전체 이름이 같으면 팬아웃과 무관하게 잇는다(그건 정말 같은 이름이다). 괄호
    조각만 상한을 받는다 — 여러 노드가 공유하는 괄호는 주석일 확률이 높다.
    """
    unique = sorted({n for n in (node_ids or ()) if isinstance(n, str) and n.strip()})
    if len(unique) < 2:
        return []

    parent = {n: n for n in unique}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)   # 결정론적 대표 선택

    # 키 맵은 **하나**여야 한다. 전체 이름과 괄호 조각을 따로 담으면 A 의 전체
    # 이름(`유방의 악성 신생물`)과 B 의 괄호 조각(`C50( 유방의 악성 신생물 )`)이
    # 서로 만나지 못한다 — 정확히 그 쌍이 이 함수를 만든 이유인데도.
    key_members: Dict[str, List[str]] = {}
    full_names: Set[str] = set()
    for node in unique:
        full = normalize_name(node)
        if full:
            full_names.add(full)
        for key in identity_keys(node):
            key_members.setdefault(key, []).append(node)

    for key, members in key_members.items():
        # 누군가의 **전체 이름**인 키는 팬아웃 상한을 면제한다: 다섯 노드가 같은
        # 이름을 쓴다면 그건 주석이 아니라 정말 같은 이름이다. 아무의 전체 이름도
        # 아닌 괄호 조각만 상한을 받는다("(제1항)" 류의 주석 방어).
        if key not in full_names and len(members) > max_fragment_fanout:
            continue
        for other in members[1:]:
            union(members[0], other)

    grouped: Dict[str, List[str]] = {}
    for node in unique:
        grouped.setdefault(find(node), []).append(node)
    return sorted((sorted(m) for m in grouped.values() if len(m) > 1),
                  key=lambda m: m[0])


def evidence_gaps(node_ids: Iterable[str],
                  chunks: Sequence[Any],
                  min_chunk_len: int = DEFAULT_MIN_CHUNK_LEN) -> Dict[str, Any]:
    """근거 사슬의 구멍 — 양쪽에서 본다.

    노드 쪽(orphan)만 보면 "그래프는 멀쩡한데 왜 인용이 안 되나"를 못 푼다.
    청크 쪽(unlinked)이 있어야 **추출이 어디서 멈췄는지**가 보인다.

    dangling_node_refs 는 반대 방향의 구멍이다: 청크는 가리키는데 그래프엔 없는
    노드 — 삭제·개명의 흔적이고, 그 인용은 클릭하면 빈 화면이 된다.
    """
    graph_nodes = {n for n in (node_ids or ()) if isinstance(n, str) and n.strip()}
    linked: Set[str] = set()
    unlinked: List[str] = []
    trivial: List[str] = []
    total = 0
    for chunk in chunks or ():
        total += 1
        refs = [r for r in (getattr(chunk, "node_ids", None) or ())
                if isinstance(r, str) and r.strip()]
        if refs:
            linked.update(refs)
            continue
        cid = str(getattr(chunk, "chunk_id", "") or "")
        text = getattr(chunk, "text", "") or ""
        (unlinked if len(text) >= min_chunk_len else trivial).append(cid)

    linked_chunks = total - len(unlinked) - len(trivial)
    return {
        "nodes": len(graph_nodes),
        "chunks": total,
        "linked_chunks": linked_chunks,
        "orphan_nodes": sorted(graph_nodes - linked),
        "unlinked_chunks": unlinked,
        "unlinked_trivial": trivial,
        "dangling_node_refs": sorted(linked - graph_nodes),
    }


def health_report(node_ids: Iterable[str],
                  chunks: Sequence[Any],
                  sample: int = 20,
                  min_chunk_len: int = DEFAULT_MIN_CHUNK_LEN,
                  max_fragment_fanout: int = DEFAULT_MAX_FRAGMENT_FANOUT,
                  ) -> Dict[str, Any]:
    """진단 요약 — 개수는 전부, 목록은 표본만.

    개수를 자르지 않는 이유: 표본만 보면 "20건"으로 보이는 것이 실제로는 340건일
    수 있다. 조용한 절단은 '다 봤다'로 읽힌다.

    extraction_coverage 가 청크 0 일 때 None 인 이유: 0/0 을 1.0 으로 보고하면
    빈 네임스페이스가 '완벽히 건강함'으로 보인다.
    """
    nodes = list(node_ids or ())
    gaps = evidence_gaps(nodes, chunks, min_chunk_len=min_chunk_len)
    clusters = duplicate_clusters(nodes, max_fragment_fanout=max_fragment_fanout)
    total_chunks = gaps["chunks"]
    coverage = (gaps["linked_chunks"] / total_chunks) if total_chunks else None
    return {
        "nodes": gaps["nodes"],
        "chunks": total_chunks,
        "linked_chunks": gaps["linked_chunks"],
        "extraction_coverage": coverage,
        "orphan_node_count": len(gaps["orphan_nodes"]),
        "orphan_node_sample": gaps["orphan_nodes"][:sample],
        "unlinked_chunk_count": len(gaps["unlinked_chunks"]),
        "unlinked_chunk_sample": gaps["unlinked_chunks"][:sample],
        "unlinked_trivial_count": len(gaps["unlinked_trivial"]),
        "dangling_node_ref_count": len(gaps["dangling_node_refs"]),
        "dangling_node_ref_sample": gaps["dangling_node_refs"][:sample],
        "duplicate_cluster_count": len(clusters),
        "duplicate_clusters": clusters[:sample],
        # 못 잡는 것을 적어 둔다 — 리포트가 깨끗하다고 그래프가 깨끗한 게 아니다.
        "limitations": [
            "표기가 전혀 다른 동의어(예: 유방암 ↔ 유방의 악성 신생물)는 "
            "구조적 정규화로 잡히지 않는다 — 임베더 기반 별도 층이 필요하다.",
        ],
    }
