"""호환 경로 — 정본은 `logosai.orchestration.progress_streamer` (2026-10-04 이전).

재export 가 아니라 별칭이다: 이 모듈 이름 자체가 정본 모듈 객체를 가리킨다.
그래야 싱글턴·isinstance·옛 경로로 한 패치가 하나로 유지된다
(재export 는 셋 다 깨졌다 — tests/test_orchestrator_compat_alias.py).
이 경로를 쓰려면 `logosai` extra 가 필요하다 (pip install logosai-ontology[logosai]).
"""
import sys as _sys

__compat_alias__ = "logosai.orchestration.progress_streamer"

try:
    from logosai.orchestration import progress_streamer as _impl
except ModuleNotFoundError as _e:
    # logosai 자체가 없거나 orchestration 이전 버전일 때만 바꿔 말한다.
    # 그 안의 다른 의존성 누락은 원래 오류 그대로 올린다 (둔갑 금지).
    if _e.name not in ("logosai", "logosai.orchestration"):
        raise
    raise ImportError(
        "ontology.orchestrator 의 실행부는 logosai.orchestration 으로 옮겨졌다 — "
        "설치된 logosai 에 그 패키지가 없다. logosai 를 orchestration 이 들어간 "
        "버전으로 올릴 것 (pip install -U logosai / logosai-ontology[logosai])."
    ) from _e

# 파일 경로로 직접 로드되면(spec_from_file_location) import 체계가 아래 바꿔치기를
# 반영하지 않고 이 모듈 자신을 돌려준다. 그때도 공개 이름은 정본 객체이게 채운다.
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
_sys.modules[__name__] = _impl
