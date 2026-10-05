#!/usr/bin/env python3
"""검색 품질 측정 하네스 (축 5, item 3).

**골든셋을 지어내지 않는다.** 두 종류의 측정을 구분한다:

  1) known-item 검색 (자동, 라벨 불필요) — 실제 노드의 이름으로 질의해 그 노드가
     상위에 오는가. recall@1/@5·MRR 로 검색의 **소여(soundness)**를 잰다.
     이건 정직한 자동 측정이다: 정답이 "그 노드 자신"으로 자명하다.
  2) 도메인 관련도 튜닝 (수동, 골든셋 필요) — "이 질의에 어떤 노드가 관련 있나"는
     사람 라벨이 있어야 한다. RRF·가중치 실측 튜닝은 이 골든셋을 전제로 하며,
     이 스크립트는 그 골든셋 JSONL 을 읽어 같은 지표를 계산할 수 있다(있을 때).

사용:
  python scripts/eval_search.py heritage_us              # known-item 자동 측정
  python scripts/eval_search.py heritage_us --n 100      # 표본 수
  python scripts/eval_search.py heritage_us --golden goldens/heritage_us.jsonl
      # 골든셋(각 줄 {"q":..., "expect":[node_id,...]}) 로 측정

환경: ONTOLOGY_ES_URL (검색 대상 ES). PYTHONPATH 는 start.sh 와 동일히 필요.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ontology.core.object_index import ObjectIndex  # noqa: E402
from ontology.engines.knowledge_graph_clean import get_knowledge_graph_engine  # noqa: E402


def _rank_of(items, expect_ids):
    """상위 결과에서 정답(expect_ids 중 아무거나)의 1-기반 순위. 없으면 None."""
    for i, it in enumerate(items):
        if it.get("node_id") in expect_ids:
            return i + 1
    return None


def _metrics(ranks, n):
    hit1 = sum(1 for r in ranks if r == 1)
    hit5 = sum(1 for r in ranks if r and r <= 5)
    mrr = sum((1.0 / r) for r in ranks if r) / n if n else 0.0
    found = sum(1 for r in ranks if r)
    return {"queries": n, "recall@1": hit1 / n, "recall@5": hit5 / n,
            "MRR": mrr, "found": found}


def known_item_goldens(namespace, n):
    """실제 노드에서 known-item 골든셋 생성 — 질의=이름, 정답=그 노드.
    결정론적 표본(node_id 정렬 후 균등 간격) — 재현 가능, 무작위 아님."""
    g = get_knowledge_graph_engine(namespace).graph
    named = sorted((nid, a.get("name") or nid) for nid, a in g.nodes(data=True)
                   if (a.get("name") or "").strip())
    if not named:
        return []
    step = max(1, len(named) // n)
    sampled = named[::step][:n]
    return [{"q": name, "expect": [nid]} for nid, name in sampled]


def run(namespace, goldens, top_k=10):
    idx = ObjectIndex(namespace)
    if not idx.available():
        print(f"❌ ES 불가(ONTOLOGY_ES_URL) — 측정 불가"); return None
    if not idx.client.indices.exists(index=idx.index):
        print(f"❌ 인덱스 없음: {idx.index} (먼저 migrate/투영 필요)"); return None
    ranks = []
    for gold in goldens:
        res = idx.search(q=gold["q"], top_k=top_k)
        ranks.append(_rank_of(res["items"], set(gold["expect"])))
    return _metrics(ranks, len(goldens))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("namespace")
    ap.add_argument("--n", type=int, default=100, help="known-item 표본 수")
    ap.add_argument("--golden", help="골든셋 JSONL 경로(있으면 이걸로 측정)")
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    if args.golden:
        with open(args.golden, encoding="utf-8") as f:
            goldens = [json.loads(l) for l in f if l.strip()]
        mode = f"golden({args.golden})"
    else:
        goldens = known_item_goldens(args.namespace, args.n)
        mode = "known-item(자동)"

    if not goldens:
        print("측정할 질의가 없다."); return
    m = run(args.namespace, goldens, top_k=args.top_k)
    if m is None:
        return
    print(f"\n검색 품질 — {args.namespace} · {mode}")
    print(f"  질의 수      : {m['queries']}")
    print(f"  recall@1     : {m['recall@1']:.3f}")
    print(f"  recall@5     : {m['recall@5']:.3f}")
    print(f"  MRR          : {m['MRR']:.3f}")
    print(f"  발견/전체    : {m['found']}/{m['queries']}")
    if not args.golden:
        print("\n※ known-item 은 검색 소여(정답이 자명한 이름 조회)를 잰다.")
        print("  도메인 관련도 튜닝(RRF·가중치)은 사람이 라벨한 골든셋이 있어야 한다 —")
        print("  --golden 으로 그 파일을 주면 같은 지표로 측정한다.")


if __name__ == "__main__":
    main()
