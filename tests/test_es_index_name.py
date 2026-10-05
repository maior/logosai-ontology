"""ES 인덱스명 소문자 계약 (2026-07-23).

배경: ES 인덱스명은 소문자만 허용한다. 대문자 네임스페이스(AI-Coach, PROJ-A)를
그대로 쓰면 `ontology-obj-AI-Coach` 가 400 invalid_index_name 으로 거부되어
노드 투영·ES 하이브리드 검색이 조용히 죽는다(실측: reindex 시 es_nodes=null).
인덱스명만 소문자로 파생하고 네임스페이스 정체성은 원형 유지한다.

ES 연결 불필요 — .index 속성만 검사(순수).
"""
from ontology.core.es_backend import ElasticsearchBackend
from ontology.core.object_index import ObjectIndex


def test_object_index_name_lowercased_for_uppercase_namespace():
    idx = ObjectIndex("AI-Coach")
    assert idx.index == idx.index.lower(), idx.index
    assert idx.index == "ontology-obj-ai-coach", idx.index
    # 네임스페이스 정체성은 원형 유지 (graph/chunk_store 는 이걸 쓴다)
    assert idx.namespace == "AI-Coach"


def test_es_backend_index_name_lowercased():
    b = ElasticsearchBackend(namespace="PROJ-A", auto_default=False)
    assert b.index == b.index.lower(), b.index
    assert b.index.endswith("-proj-a"), b.index


def test_lowercase_namespace_unchanged():
    # 이미 소문자면 그대로 (회귀: 기존 네임스페이스 인덱스명 불변)
    assert ObjectIndex("ins_cancer_demo").index == "ontology-obj-ins_cancer_demo"


if __name__ == "__main__":
    import sys
    failed = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_") and callable(fn):
            try:
                fn(); print(f"  PASS {n}")
            except AssertionError as e:
                failed += 1; print(f"  FAIL {n}: {e}")
    print("OK" if not failed else f"FAILED {failed}")
    sys.exit(1 if failed else 0)
