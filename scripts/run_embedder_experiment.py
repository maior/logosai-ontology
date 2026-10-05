"""Tier 1 임베더 실험 — 스크래치 네임스페이스에서만 (Phase 4).

임베더 교체는 재색인(쓰기)을 동반하므로 라이브 네임스페이스에서 돌리지
않는다. 절차 (P-5 에서 검증된 스크래치 NS 기법 + PG=진실 운영 규율):

  1. 원본 NS 의 청크·그래프·골든셋 파일을 `_exp_` 접두 스크래치 NS 로 복사
  2. 임베더 env 를 바꾼 **별도 프로세스**로 서버 없이 재색인 + Tier 0 실행
     (라이브 서버는 무접촉 — 스크래치 파일만 만진다)
  3. 결과는 원본 NS 의 실험 스토어에 layer=retrieval, axes.embedder 로 기록
  4. 스크래치 파일 삭제

사용:
  .venv/bin/python ontology/scripts/run_embedder_experiment.py \
      --namespace ins_cancer_demo --models BAAI/bge-m3 jhgan/ko-sroberta-nli

주의: 모델당 전체 재색인 1회 — 191노드 기준 bge-m3 ~28s 실측. 라이브 API
가 아니라 파일 복사인 이유: 스크래치 NS 는 PG 대상이 아니어서(허용목록 밖)
파일이 곧 진실이다 — PG-하이드레이트 덮어쓰기 사고의 전제가 성립하지 않는다.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # ontology/
DATA = ROOT / "data"
PY = sys.executable

WORKER = r"""
import json, sys
ns, out_path = sys.argv[1], sys.argv[2]
from ontology.server.service import OntologyBuilderService
from ontology.engines.knowledge_graph_clean import get_knowledge_graph_engine
svc = OntologyBuilderService()
svc._namespace_exists = lambda n: True          # 스크래치 — 목록 밖
get_knowledge_graph_engine(ns).rebuild_semantic_index()
res = svc.run_retrieval_experiments(
    ns, statuses=["confirmed", "verified"], actor="embedder_experiment")
json.dump(res, open(out_path, "w"), ensure_ascii=False)
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--keep-scratch", action="store_true")
    args = ap.parse_args()

    src = args.namespace
    if src.startswith("_exp_"):
        raise SystemExit("원본 네임스페이스를 지정하라 (스크래치 아님)")

    files = [f"chunks_{src}.jsonl", f"kg_{src}.json", f"goldenset_{src}.jsonl"]
    missing = [f for f in files if not (DATA / f).exists()]
    if missing:
        raise SystemExit(f"원본 파일 없음: {missing}")

    from ontology.core.experiment import get_experiment_store
    dest_store = get_experiment_store(src)

    for model in args.models:
        scratch = f"_exp_{src}"
        for f in files:
            shutil.copy(DATA / f, DATA / f.replace(src, scratch))
        out = DATA / f"_exp_result_{src}.json"
        env = {**os.environ,
               "ONTOLOGY_EMBEDDING_MODEL": model,
               "PYTHONPATH": str(ROOT.parent)}
        print(f"→ {model}: 재색인 + Tier 0 (스크래치 {scratch})")
        subprocess.run([PY, "-c", WORKER, scratch, str(out)],
                       check=True, env=env, cwd=str(ROOT.parent))
        import json
        res = json.load(open(out))
        for rec in res.get("records", []):
            rec["axes"]["embedder"] = model
            rec["config"]["node_model"] = model
            rec["namespace"] = src
            dest_store.record(rec)
        print(f"   기록 {len(res.get('records', []))}건 → experiments_{src}.jsonl")
        if not args.keep_scratch:
            for f in files:
                (DATA / f.replace(src, scratch)).unlink(missing_ok=True)
            for extra in DATA.glob(f"*_{scratch}.*"):
                extra.unlink(missing_ok=True)
            out.unlink(missing_ok=True)

    print("완료 — 파레토는 GET /experiments/recommendation 또는 실험 탭에서")


if __name__ == "__main__":
    main()
