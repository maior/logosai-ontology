"""인제스트 미리보기 설정 — '벡터화 전 교감'용 순수 함수.

벡터화(수집)는 준-비가역·유상 작업이다. 사용자가 '무엇으로 벡터화되는가'를
모른 채 수집 버튼을 누르지 않도록, 감식(analyze) 응답에 이 설정을 실어
확인 단계에서 보여준다:
  · 임베딩 모델 (청크 의미검색용)
  · 청크 설정 (크기·겹침·분할 방식; 약관 등 조문형은 조 단위 heading)
  · 키워드 검색(ES BM25 + dense 하이브리드) 가능 여부
  · 생성될 인덱스명 (object / chunk) — 네임스페이스로 매개화

인덱스명은 네임스페이스에 종속되지만 감식 시점엔 ns 가 미정이므로 패턴
('{namespace}')으로 두고, 프론트가 사용자가 입력한 ns 로 치환한다.
es_available / embedder_available 은 호출부(service)가 cheap 하게 판정해 주입
— 이 함수는 순수하게 유지(라이브 서버 없이 테스트 가능)한다.
"""

import os
from typing import Any, Dict, Optional

from .object_index import OBJ_PREFIX
from .semantic_index import DEFAULT_EMBEDDING_MODEL

DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 120
DEFAULT_SEGMENT_MODE = "auto"


def build_ingest_config(
    es_available: bool,
    embedder_available: bool,
    namespace: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    segment_mode: str = DEFAULT_SEGMENT_MODE,
) -> Dict[str, Any]:
    """수집 전 사용자에게 보여줄 설정 payload (순수 함수)."""
    ns = namespace or "{namespace}"
    return {
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "embedder_available": bool(embedder_available),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "segment_mode": segment_mode,
        "es_available": bool(es_available),
        # ES 가 있어야 키워드(BM25) + 벡터 하이브리드가 켜진다. 없으면 의미검색만.
        "keyword_search": bool(es_available),
        "vector_backend": os.environ.get("ONTOLOGY_VECTOR_BACKEND", "auto"),
        "object_index": f"{OBJ_PREFIX}-{ns}",
        "chunk_index": f"chunks_{ns}",
        "extraction_modes": ["exhaustive", "topic"],
    }
