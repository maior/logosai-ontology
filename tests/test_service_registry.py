"""서비스 정본 표 등록 계약 (2026-08-21).

`scripts/services.sh` 는 이 모노레포의 **유일한 서비스 목록**이다 — 기동·종료·
상태·자동재기동 네 곳이 이 표 하나를 읽는다. 표에 없는 것은 아무도 띄우지
않고 아무도 감시하지 않는다 (2026-08-17 FORGE 5일 18시간 정전이 정확히
"어떤 목록에도 없었다"였다).

온톨로지 3종(9274 서버 · 9275 콘솔 · 8915 grounded 에이전트 ACP)은 그
사고와 **같은 구멍**에 있었다. 8915 는 기동 커맨드가 코드베이스 어디에도
없어 `.claude/settings.local.json` 의 퍼미션 항목에만 남아 있었고, 실제로
2026-08-21 재구동 때 사람이 손으로 env 를 조립하다 `ACP_AGENTS_JSON` 을
빠뜨려 **에이전트 0개로 조용히 떴다** (서버는 "시작 성공"을 보고했다).

계약:
  1. 표의 모든 항목은 실존하고 실행 가능한 start/stop 스크립트를 가리킨다.
  2. 포트는 유일하다 (두 서비스가 같은 포트면 상태 판정이 서로를 가린다).
  3. 온톨로지 3종이 표에 있고, 9274 가 8915 보다 **먼저** 온다 —
     8915 의 에이전트는 9274 의 REST 를 호출하므로 의존 순서가 있다.
  4. `ontology/scripts/` 의 모든 기동 스크립트는 표에 실려 있다 —
     "스크립트는 있는데 아무도 부르지 않는" 상태의 재발 방지.

이 리포는 단독 체크아웃(public wheel)으로도 쓰이므로, 모노레포 루트가
없으면 **건너뛴다** — 없는 파일을 실패로 세면 공개 리포의 CI 가 깨진다.
"""

import os
import re
from pathlib import Path

import pytest

_ONTOLOGY_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _ONTOLOGY_ROOT.parent
_SERVICES_SH = _REPO_ROOT / "scripts" / "services.sh"

pytestmark = pytest.mark.skipif(
    not _SERVICES_SH.exists(),
    reason="모노레포 루트 없음 (온톨로지 단독 체크아웃) — 표 계약은 여기 없다")


def _entries():
    """LOGOS_SERVICES 배열의 항목을 (id, port, health, start, stop) 로 판다.

    bash 를 실행하지 않고 텍스트로 읽는다 — 테스트가 서비스를 띄우면 안 된다.
    """
    text = _SERVICES_SH.read_text(encoding="utf-8")
    m = re.search(r"LOGOS_SERVICES=\((.*?)\n\)", text, re.S)
    assert m, "LOGOS_SERVICES 배열을 찾지 못했다"
    out = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.strip('"').split("|")
        assert len(parts) == 5, f"형식 위반 (id|port|health|start|stop): {line}"
        out.append(tuple(p.strip() for p in parts))
    assert out, "표가 비어 있다"
    return out


def _script_path(cmd: str) -> Path:
    """기동 명령에서 스크립트 경로만 — 인자(`--bg` 등)는 떼어낸다."""
    return _REPO_ROOT / cmd.split()[0]


class TestRegistryIntegrity:
    def test_scripts_exist_and_are_executable(self):
        broken = []
        for sid, _port, _health, start, stop in _entries():
            for label, cmd in (("start", start), ("stop", stop)):
                path = _script_path(cmd)
                if not path.exists():
                    broken.append(f"{sid} {label}: 없음 {path}")
                elif not os.access(path, os.X_OK):
                    broken.append(f"{sid} {label}: 실행권한 없음 {path}")
        assert not broken, "표가 없는/못 도는 스크립트를 가리킨다:\n" + "\n".join(broken)

    def test_ports_are_unique(self):
        ports = [e[1] for e in _entries()]
        dupes = {p for p in ports if ports.count(p) > 1}
        assert not dupes, f"포트 중복 — 상태 판정이 서로를 가린다: {dupes}"


class TestOntologyServicesRegistered:
    """온톨로지 3종이 표에 있는가 — 이 파일이 생긴 이유."""

    def test_all_three_ports_present(self):
        ports = {e[1] for e in _entries()}
        missing = {"9274", "9275", "8915"} - ports
        assert not missing, (
            f"온톨로지 서비스가 정본 표에 없다: {missing}. "
            "표에 없으면 start_all/keepalive 가 모른다 — FORGE 정전과 같은 구멍")

    def test_server_starts_before_agent_acp(self):
        """8915 의 에이전트는 9274 REST 를 호출한다 — 순서가 있다."""
        ports = [e[1] for e in _entries()]
        assert ports.index("9274") < ports.index("8915"), (
            "9274(온톨로지 서버)가 8915(에이전트 ACP)보다 먼저 와야 한다 — "
            "배열 순서가 곧 기동 순서다")

    def test_every_ontology_start_script_is_registered(self):
        """스크립트는 있는데 아무도 부르지 않는 상태의 재발 방지."""
        registered = " ".join(e[3] for e in _entries())
        orphans = [p.name for p in sorted((_ONTOLOGY_ROOT / "scripts").glob("start*.sh"))
                   if f"ontology/scripts/{p.name}" not in registered]
        assert not orphans, (
            f"기동 스크립트가 정본 표에 없다: {orphans} — "
            "아무도 띄우지 않고 아무도 감시하지 않는다")
