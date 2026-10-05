"""GNN+RL 임베더 교체 계약 (Task #11, 2026-07-15).

배경: paraphrase-multilingual-MiniLM 이 한국어에서 퇴화 (무관 문장 cosine 0.96,
2026-07-02 실측) → GNN+RL 신뢰도 항상 저조 → 채택 0%. semantic_index 는 이미
jhgan/ko-sroberta-nli 로 교체됨 — ml 층도 동일 모델로 정렬한다 (프로세스 내 공유).

계약: ① 기본 모델 = ko-sroberta (env ONTOLOGY_ML_EMBEDDING_MODEL 오버라이드)
② 차원 자동 정합 — query 768, state = query+graph+history (불변식)
③ env 로 구 모델 지정 시 384/512 로 복귀 (롤백 경로)

직접 실행: .venv/bin/python ontology/ml/test_ml_embedding_swap.py
"""

import importlib
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ontology/__init__ 의 무거운 의존성 체인 회피 — ml 패키지만 직접 로드
sys.path.insert(0, os.path.join(_ROOT, "ontology"))


def _fresh_config():
    import ml.config as cfg
    importlib.reload(cfg)
    return cfg


def main():
    fails = []

    def t(name, cond):
        print(("PASS  " if cond else "FAIL  ") + name)
        if not cond:
            fails.append(name)

    # M-1 기본: ko-sroberta + 768 + state 896
    os.environ.pop("ONTOLOGY_ML_EMBEDDING_MODEL", None)
    os.environ.pop("ONTOLOGY_ML_QUERY_DIM", None)
    cfg = _fresh_config()
    c = cfg.SelectorConfig()
    t("M-1 기본 모델 = jhgan/ko-sroberta-nli", "ko-sroberta" in c.embedding_model)
    t("M-2 query_embedding_dim = 768", c.rl.query_embedding_dim == 768)
    t("M-3 state_dim 자동 정합 (768+64+64=896)",
      c.rl.state_dim == c.rl.query_embedding_dim + c.rl.graph_embedding_dim + c.rl.history_dim == 896)

    # M-4 env 오버라이드 (롤백 경로: 구 MiniLM 384)
    os.environ["ONTOLOGY_ML_EMBEDDING_MODEL"] = "paraphrase-multilingual-MiniLM-L12-v2"
    os.environ["ONTOLOGY_ML_QUERY_DIM"] = "384"
    cfg = _fresh_config()
    c2 = cfg.SelectorConfig()
    t("M-4 env 오버라이드로 구 모델 복귀", "MiniLM" in c2.embedding_model
      and c2.rl.query_embedding_dim == 384 and c2.rl.state_dim == 512)

    # M-5 dim 미지정 + 알려진 모델명 → 자동 추론
    os.environ["ONTOLOGY_ML_EMBEDDING_MODEL"] = "jhgan/ko-sroberta-multitask"
    os.environ.pop("ONTOLOGY_ML_QUERY_DIM", None)
    cfg = _fresh_config()
    c3 = cfg.SelectorConfig()
    t("M-5 sroberta 계열 자동 768", c3.rl.query_embedding_dim == 768)

    # 정리
    os.environ.pop("ONTOLOGY_ML_EMBEDDING_MODEL", None)
    os.environ.pop("ONTOLOGY_ML_QUERY_DIM", None)

    print("\nRESULT:", "GREEN" if not fails else f"RED ({len(fails)} failing)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
