#!/usr/bin/env python3
"""계획 회귀 하네스 — 실제 LLM 플래너를 고정된 조건에서 반복 실행한다.

  python scripts/plan_regression.py record  [-n 8] [--with-selector] [--concurrency 4]
  python scripts/plan_regression.py compare [-n 5] [--with-selector] [--concurrency 4]

조건을 고정하는 이유: 레지스트리가 바뀌면(에이전트 추가·설명 수정) 계획도 바뀐다 —
그건 플래너 회귀가 아니다. 그래서 레지스트리는 스냅샷 파일(운영 계획 출처인 logos_api
DB 레지스트리)로 고정하고, 그 지문을 기준선에 남긴다. 하이브리드 선택기는 학습 상태가
변하고 디스크에 기록도 남겨 재현성이 없으므로 기본으로 끈다 — 이 하네스는 프롬프트
조립과 LLM 호출(P2 가 바꾸는 부분)을 격리해 잰다.

경로는 운영과 같다: ontology.orchestrator.QueryPlanner (logos_api 가 쓰는 import 경로).
관측 DB 를 오염시키지 않도록 Pulse 전송은 끈다.

판정 로직: ontology/orchestrator/plan_regression.py (순수, 테스트됨).
종료 코드: compare 에서 실패(drift·missing)가 있으면 1. 안정 시나리오(기준선 N≥8, 최빈 ≥7/8)만
엄격히 판정하고, 흔들리는 시나리오는 정보로 보고한다 — 판정 규칙은 plan_regression.compare.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("LOGOS_PULSE_DISABLED", "1")

HERE = Path(__file__).resolve().parent / "plan_regression"
SCENARIOS = HERE / "scenarios.json"
SNAPSHOT = HERE / "registry_snapshot.json"
BASELINE = HERE / "baseline.json"


def _registry(agents):
    from ontology.orchestrator import AgentRegistry
    from ontology.orchestrator.models import AgentRegistryEntry, AgentSchema

    # 운영 모양: Logos 기본 11개를 먼저, DB 에이전트를 나중에 (logos_api orchestrator_service).
    # 나열 순서·내용이 라우팅에 영향을 준다 — 기본값을 빼거나 그중 블록 하나만 빼도 안정 시나리오 6개가 회귀했다 (2026-10-05).
    from ontology.orchestrator.logos_agents import logos_default_agents
    reg = AgentRegistry(defaults=logos_default_agents())
    for a in agents:
        reg.register_agent(AgentRegistryEntry(
            agent_id=a["agent_id"], name=a.get("name") or a["agent_id"],
            description=a.get("description") or "", capabilities=a.get("capabilities") or [],
            tags=a.get("tags") or [], schema=AgentSchema(input_type="query", output_type="text")))
    return reg


async def _run(runs: int, with_selector: bool, concurrency: int, only=None):
    from ontology.orchestrator import QueryPlanner
    from ontology.orchestrator.plan_regression import (
        at_level, plan_shape, registry_fingerprint, summarize,
    )

    agents = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["agents"]
    scenarios = json.loads(SCENARIOS.read_text(encoding="utf-8"))["scenarios"]
    if only:
        scenarios = [s for s in scenarios if s["id"] in only]
    registry = _registry(agents)
    QueryPlanner.USE_HYBRID_SELECTOR = with_selector
    gate = asyncio.Semaphore(concurrency)

    async def once(query):
        async with gate:
            try:
                return plan_shape(await QueryPlanner(registry=registry).create_plan(query, {}))
            except Exception as e:   # noqa: BLE001 — 예외도 계획의 한 모양으로 기록한다
                return plan_shape(e)

    started = time.time()
    shapes = await asyncio.gather(*[once(s["query"]) for s in scenarios for _ in range(runs)])
    out, i = {}, 0
    for s in scenarios:
        raw = shapes[i:i + runs]
        level = s.get("compare", "plan")
        out[s["id"]] = {**summarize(at_level(x, level) for x in raw),
                        "level": level, "raw_counts": summarize(raw)["counts"]}
        i += runs
    return {
        "config": {"registry": registry_fingerprint(agents), "selector": with_selector,
                   "runs": runs, "model": QueryPlanner.MODEL,
                   "temperature": QueryPlanner.TEMPERATURE},
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - started, 1),
        "scenarios": out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("record", "compare"))
    ap.add_argument("-n", "--runs", type=int, default=None,
                    help="반복 횟수 (기본: record 8, compare 5)")
    ap.add_argument("--with-selector", action="store_true")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", type=Path, help="compare 결과를 저장할 경로")
    args = ap.parse_args()

    runs = args.runs or (8 if args.mode == "record" else 5)
    result = asyncio.run(_run(runs, args.with_selector, args.concurrency, args.only))

    if args.mode == "record":
        BASELINE.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        from ontology.orchestrator.plan_regression import MIN_STABLE_RUNS, STABLE_SHARE
        stable = [k for k, v in result["scenarios"].items()
                  if v["n"] >= MIN_STABLE_RUNS and v["counts"][v["modal"]] / v["n"] >= STABLE_SHARE]
        print(f"기준선 저장: {BASELINE} — 시나리오 {len(result['scenarios'])}개 × {runs}회, "
              f"{result['elapsed_s']}s, 안정(판정 대상) {len(stable)}개, "
              f"흔들림(정보) {len(result['scenarios']) - len(stable)}개")
        return 0

    from ontology.orchestrator.plan_regression import FAILING, compare
    findings = compare(json.loads(BASELINE.read_text(encoding="utf-8")), result)
    if args.out:
        args.out.write_text(json.dumps({"result": result, "findings": findings},
                                       ensure_ascii=False, indent=1), encoding="utf-8")
    bad = [f for f in findings if f["status"] in FAILING]
    for f in findings:
        mark = {"same": "  ", "variable": "~ ", "shifted": "! "}.get(f["status"], "✗ ")
        line = f"{mark}{f['id']:4} {f['status']:18} {f.get('baseline')}"
        if f["status"] != "same":
            line += f"  →  {f.get('current')}  {f.get('current_counts', '')}"
        print(line)
    count = lambda st: sum(f["status"] == st for f in findings)  # noqa: E731
    print(f"\n시나리오 {len(findings)}개: 같음 {count('same')}, 흔들림 {count('variable')}, "
          f"눈여겨볼 변화 {count('shifted')}, 회귀 {len(bad)} ({result['elapsed_s']}s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
