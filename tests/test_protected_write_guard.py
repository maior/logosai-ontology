"""보호 네임스페이스 쓰기 가드 계약 (2026-08-21).

`PROTECTED_NAMESPACES`(현재 `default` = 에이전트 라우팅 그래프)는 관리 화면의
어떤 실수도 넘지 못하는 선이다. 그런데 가드가 데코레이터가 아니라 **각
엔드포인트에 손으로 한 줄씩** 삽입돼 있어, 빠뜨려도 아무도 모른다.

실측(2026-08-21): `POST /review/reject` 가 그 상태였다 — 이 경로는 노드를
그래프에서 **영구 제거**하고(엣지·PG·시맨틱 색인까지) 묘비를 남긴다.
`DELETE /graphs/{ns}/edges`·`nodes/merge`·`nodes/rename` 은 전부 403 인데
정작 노드 통삭제만 뚫려 있었다. 정책 누락이지 설계 선택이 아니다.

이 파일은 **파괴적 쓰기 경로**가 보호선을 지키는지 고정한다. 여기 목록에
없는 새 경로가 생기면 그건 다음 사람이 추가할 몫이지만, 적어도 이미 아는
경로가 조용히 풀리는 일은 막는다.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ontology.core.review_store import reset_review_stores

API = "/api/v1/ontology"
PROTECTED = "default"


@pytest.fixture()
def client(tmp_path):
    from ontology.server import router as server_router
    from ontology.server.service import OntologyBuilderService

    reset_review_stores()
    service = OntologyBuilderService(data_dir=tmp_path)
    app = FastAPI()
    app.include_router(server_router.router, prefix=API)
    app.dependency_overrides[server_router.get_ontology_service] = lambda: service
    yield TestClient(app)
    reset_review_stores()


# (라벨, 메서드, 경로, body) — 전부 보호 네임스페이스를 파괴하거나
# 판정을 새기는 경로다. 가드가 body 검증보다 **먼저** 걸려야 하므로
# body 는 최소만 채운다.
DESTRUCTIVE = [
    ("노드 거절(=그래프에서 제거)", "post", "/review/reject", {"node_id": "T:x"}),
    ("노드 확정(판정 기록)", "post", "/review/confirm", {"node_id": "T:x"}),
    ("노드 병합", "post", "/nodes/merge", {"winner": "T:a", "losers": ["T:b"]}),
    ("노드 재분류", "post", "/nodes/rename", {"node_id": "T:a", "new_type": "U"}),
    ("엣지 추가", "post", "/edges",
     {"source": "T:a", "predicate": "p", "target": "T:b"}),
    ("노드 생성", "post", "/nodes", {"node_type": "T", "name": "x"}),
]


@pytest.mark.parametrize("label,method,path,body", DESTRUCTIVE,
                         ids=[d[0] for d in DESTRUCTIVE])
def test_protected_namespace_rejects_destructive_write(client, label, method,
                                                       path, body):
    resp = getattr(client, method)(f"{API}/graphs/{PROTECTED}{path}", json=body)
    assert resp.status_code == 403, (
        f"{label}: 보호 네임스페이스에 {resp.status_code} 로 통과했다 — "
        f"가드 누락 (본문: {resp.text[:120]})")


def test_unprotected_namespace_is_not_blocked_by_the_guard(client):
    """가드가 모든 것을 막으면 기능이 죽는다 — 보호 대상만 막아야 한다.

    404(네임스페이스 없음)든 400이든 상관없다. **403 이 아니면** 된다.
    """
    resp = client.post(f"{API}/graphs/보통네임스페이스/review/reject",
                       json={"node_id": "T:x"})
    assert resp.status_code != 403, "보호 대상이 아닌데 가드가 막았다"
