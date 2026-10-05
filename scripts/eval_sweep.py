#!/usr/bin/env python3
"""검색 knob 스윕 (Phase 2-C) — 골든셋으로 entry_ratio 등을 나란히 재서
'어느 knob 이 어느 시나리오 유형을 얼마나 움직이나'를 본다.

evaluate_cases 는 채널을 (query, top_k)→[node_id] 함수로 받는 범용 구조라,
설정만 다른 GraphConditionedRetriever 를 여러 채널로 넣어 비교하는 게 목적
(search_qa 도크스트링의 계약). retrieve 채널의 노드 랭킹은 entry-node 우선
어댑터(retrieve_result_to_nodes)로 서비스와 동일하게 만든다.

RRF_K·BM25_WEIGHT 는 현재 하드코딩 상수라 스윕 대상이 아니다(리팩터 필요).
entry_ratio·max_terms 만 파라미터 주입 가능해 여기서 스윕한다.

사용:
  python scripts/eval_sweep.py ins_cancer_demo
  python scripts/eval_sweep.py ins_cancer_demo --k 5 --include-drafts
  python scripts/eval_sweep.py ins_cancer_demo --entry-ratios 0.3,0.5,0.7

환경: 서버와 동일(ONTOLOGY_ES_URL, 임베더). start.sh 와 같은 PYTHONPATH.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ontology.core.graph_retrieval import GraphConditionedRetriever  # noqa: E402
from ontology.core.search_qa import (evaluate_cases, get_golden_set,  # noqa: E402
                                      retrieve_result_to_nodes)
from ontology.engines.knowledge_graph_clean import get_knowledge_graph_engine  # noqa: E402


def _retrieve_channel(retriever, entry_ratio, max_terms):
    """설정이 고정된 retrieve 채널 RankFn — 서비스와 동일한 노드 랭킹."""
    def rank(query, top_k):
        exp = retriever.expand(query, entry_ratio=entry_ratio, max_terms=max_terms)
        hits = retriever.search(query, top_k=top_k,
                                entry_ratio=entry_ratio, max_terms=max_terms)
        result = {
            "expansion": {"entry_nodes": exp.entry_nodes,
                          "expanded_nodes": exp.expanded_nodes},
            "hits": [{"node_ids": list(getattr(h.chunk, "node_ids", []) or [])}
                     for h in hits],
        }
        return retrieve_result_to_nodes(result)
    return rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("namespace")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--include-drafts", action="store_true")
    ap.add_argument("--entry-ratios", default="0.3,0.5,0.7")
    ap.add_argument("--max-terms", type=int, default=8)
    args = ap.parse_args()

    golden = get_golden_set(args.namespace)
    golden.load_from_disk()
    cases = golden.cases()
    if not cases:
        print(f"골든셋 비어 있음: {args.namespace}")
        return

    engine = get_knowledge_graph_engine(args.namespace)
    retriever = GraphConditionedRetriever(namespace=args.namespace)

    channels = {
        "semantic": lambda q, k: [h["node_id"]
                                  for h in engine.semantic_search(q, top_k=k)],
    }
    ratios = [float(x) for x in args.entry_ratios.split(",") if x.strip()]
    for er in ratios:
        channels[f"er={er:g}"] = _retrieve_channel(retriever, er, args.max_terms)

    res = evaluate_cases(cases, channels, k=args.k,
                         include_drafts=args.include_drafts)

    names = list(channels)
    print(f"\n네임스페이스={args.namespace}  n={res['cases']}  k={args.k}  "
          f"(include_drafts={args.include_drafts})")
    print("\n=== 채널 종합 ===")
    print(f"  {'채널':12s} {'hit@1':>7s} {'hit@'+str(args.k):>7s} {'mrr':>7s}")
    for name in names:
        m = res["channels"][name]
        print(f"  {name:12s} {m['hit@1']:7.2f} {m['hit@'+str(args.k)]:7.2f} "
              f"{m['mrr']:7.2f}")

    print("\n=== 태그별 hit@k (knob 이 움직이는 곳을 본다) ===")
    hdr = "  " + f"{'태그':12s}" + "".join(f"{n:>10s}" for n in names) + "   n"
    print(hdr)
    for tag, ch in res.get("by_tag", {}).items():
        row = "  " + f"{tag:12s}"
        n_tag = 0
        for n in names:
            m = ch.get(n, {})
            row += f"{m.get('hit@'+str(args.k), 0.0):10.2f}"
            n_tag = m.get("cases", n_tag)
        print(row + f"  {n_tag:3d}")


if __name__ == "__main__":
    main()
