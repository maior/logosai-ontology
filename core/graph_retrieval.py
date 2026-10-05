"""
그래프-조건부 검색 (graph-conditioned retrieval).

축 4 — 온톨로지와 검색을 실제로 결합한다.

앞선 분석에서 이 자리는 비어 있었다. aicoach 는 온톨로지와 ES 를 두 개의
평행한 검색 시스템으로 두고 `source` 키워드 하나로만 이었다: 동의어
테이블(kg/match.py:21-26)은 ES 에 닿지 않고, 그래프 순회로 쿼리를 확장하는
코드도 없고, ES aggregation 은 0건이다. 결국 상품명 검색이 요약서를 잡는
문제를 `f"{label} 보장 보험금 지급사유"` 라는 **하드코딩 리터럴**로 편향시켜
막았다 (rag/search.py:126) — 온톨로지에 그 지식이 있는데도 못 쓴 것이다.
KorAct 도 substring 매처에 머물며 `logos_llm` 자리를 비워둔 채 기다렸다.

Logos 는 그래프와 임베딩을 한 곳에 가진 유일한 시스템이라 이걸 할 수 있다.

축 3 에서 실측된 실패가 이 파일의 존재 이유다 (실제 ko-sroberta):
    질의 "계약을 무를 수 있나요?"
    기대  "청약철회권은 ... 15일 이내에 행사할 수 있다."
    실제  "보험료 납입이 연체되면 계약은 실효된다."   ← "계약" 글자에 끌려감

두 채널을 융합한다:
- chunk 채널: 청크 임베딩/하이브리드 검색 (확장된 질의로)
- graph  채널: 질의 → 진입 노드(의미) → 온톨로지 확장 → 그 노드가 달린 청크

융합은 RRF(Reciprocal Rank Fusion). 두 채널의 점수는 의미가 달라(청크 코사인
vs 노드 코사인) 선형 가중합이 성립하지 않는다 — RRF 는 점수가 아니라 **순위**만
쓰므로 스케일이 달라도 안전하다. aicoach 가 한 질의 안에서 BM25 와 cosine 을
선형 블렌드한 것과는 상황이 다르다(그건 한 엔진 안의 두 점수라 튜닝이 가능했다).

확장어는 전부 그래프에서 읽는다 — 하드코딩 어휘 금지(프로젝트 절대 원칙).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from .chunk_store import StoredChunk

# RRF 표준 상수. 큰 k 는 상위 순위 간 차이를 눌러 한 채널이 결과를 독식하는 것을
# 막는다. 60 은 원 논문(Cormack et al. 2009) 이래의 관행값이고, 채널이 둘뿐이라
# 재튜닝할 근거가 없다 — 근거 없이 바꾸느니 관행을 따른다.
RRF_K = 60

# 진입 노드 게이트 — **절대 임계값을 쓰지 않는다**.
#
# 처음엔 min_entry_score=0.25 로 짰다가 두 번 연달아 정답 노드를 잘라먹었다
# (실측: 정답이 1위인데 점수가 0.18, 0.07). 코사인 점수는 모델·코퍼스마다
# 스케일이 달라 **보정되지 않은 값**이라, 절대 임계값은 정확히 이런 식으로
# 조용히 실패한다. ko-sroberta 로 튜닝한 숫자는 gemini-embedding 에서 무의미하고
# 도메인이 바뀌어도 무의미하다.
#
# 대신 두 단계로 거른다:
# 1) 절대 바닥: score > 0. 이건 지어낸 숫자가 아니라 코사인의 정의다 —
#    유사도가 0 이하인 노드는 무관하다.
# 2) 상대 게이트: 최고점 대비 비율. 척도-무관이라 임베더가 바뀌어도 성립한다.
#
# ENTRY_RATIO 0.5 는 여전히 측정값이 아니다. 다만 절대값과 달리 임베더 교체에
# 무너지지 않는다. 도메인 골든셋이 생기면 aicoach 가 BM25 가중치를 뽑아냈듯이
# 측정해서 정해야 한다.
DEFAULT_MIN_ENTRY_SCORE = 0.0
DEFAULT_ENTRY_RATIO = 0.5

# 진입 노드 상한 — **관계를 채우자 이 값이 knob 이 됐다.**
#
# `retrieve_result_to_nodes` 는 진입 노드를 전부 앞세운다(그 설계는 측정으로
# 국소 최적임이 확인됐다). 그런데 진입이 5개면 k=5 채점에서 **확장 노드가 한 칸도
# 보이지 않는다** — 확장이 정답을 찾아와도 6위부터 시작한다.
#
# 실측이 정확히 그것이었다. 손 시드 케이스 "연만기와 세만기는 무엇의 종류인가?"는
# 확장이 `via=is_a↑ InsuranceTerm:보험기간` 으로 정답을 데려오는데 k=5 에서 미검출,
# k=10 에서 **6위**였다. 진입 5개가 앞을 다 먹은 것이다.
#
#   entry_k / entry_ratio   hit@1    hit@5    MRR      (46 케이스, target=node)
#   ────────────────────────────────────────────────────────────────────────
#   3 / 0.5                 0.5217   0.8478   0.6428   ← 채택
#   2 / 0.5                 0.5217   0.8043   0.6366
#   5 / 0.9                 0.5217   0.7826   0.6286
#   5 / 0.5                 0.5217   0.7609   0.6232   ← 종전
#
# hit@1 은 전 조합에서 같다 — 1위는 언제나 진입 1위이므로 이 knob 이 닿지 않는다.
# `entry_ratio` 는 entry_k=3 에서 무관해지므로(0.5 == 0.9) 건드리지 않는다.
#
# **대가가 있다 (숨기지 않는다).** 진입을 줄이면 진입 4·5위에 있던 정답을 잃는다:
#
#   자          hit@1              hit@5              MRR
#   ──────────────────────────────────────────────────────────────
#   node        0.5217 (불변)      0.7609 → 0.8478    0.6232 → 0.6428
#   evidence    0.7174 → 0.6957    0.9783 → 1.0000    0.8442 → 0.8333
#
# evidence 의 hit@1/MRR 하락은 각각 **1건**(46 케이스에서 0.0217)이고 잡음 범위다.
# node 의 hit@5 +4건은 잡음보다 크다. 그래서 채택하되 균형이 이렇다는 것을 남긴다.
#
# 구체적 손실 사례: "상법에서 고지의무라고 부르는 것이 이 약관에서는 무엇인가?" 는
# 4위 → 미검출이 됐다. 진입 상위가 `약관`(0.713) · `목적 및 용어의 정의`(0.686) ·
# `중요한 사항`(0.663) 같은 **일반 용어**로 채워지고 정답 `계약 전 알릴 의무`(0.656)
# 와 `고지의무`(0.641)가 4·5위였다. 즉 이 knob 의 대가는 **진입 품질이 나쁠 때
# 커진다** — 그리고 그 원인은 질의 쪽 비대칭이다(semantic_index docstring 의
# "유방암" 측정 참고). 임베더를 고치면 이 대가가 줄어든다.
#
# ⚠️ **이 값은 그래프 상태에 종속적이다** — `DEFAULT_MAX_TERMS` 와 같은 부류다.
# is_a 가 0개이고 확장이 빈 목록이던 때 진입을 줄이면 후보만 줄어 손해였을 것이다.
# 관계(is_a 7 + sameAs 3)를 채운 것이 이 knob 을 열었다. 관계가 더 늘면 다시 재야
# 한다.
DEFAULT_ENTRY_K = 3

# 확장어 상한 — 확장어가 무한정 늘면 원 질의가 희석된다.
#
# ⚠️ **이 값을 8 → 2 로 바꿨다가 되돌렸다. 그 왕복이 이 주석의 요점이다.**
#
# 고아 노드 27개(근거 링크 0)를 안고 측정했을 때는 2 가 8 보다 확실히 좋았다
# (hit@1 0.6250 vs 0.5000, MRR 0.6792 vs 0.6271 — 15개 조합 중 `0.5/8` 이 꼴찌).
# 그래서 "확장어가 잡음이다"라고 읽고 2 로 낮췄다.
#
# 그 뒤 고아 노드 23개에 근거 청크 66개를 이었더니(approve_orphan_links)
# **순서가 뒤집혔다** (16 케이스, k=5, evidence):
#
#   max_terms   hit@1    hit@5    MRR
#   ──────────────────────────────────────
#   0           0.6875   0.9375   0.7969
#   1           0.6875   1.0000   0.8229
#   2           0.7500   1.0000   0.8542   ← 회복 전 최고였던 값
#   3           0.6875   1.0000   0.8042
#   6           0.6875   1.0000   0.8385
#   8           0.8125   1.0000   0.8958   ← 원래 값이 다시 최고
#   12          0.8125   1.0000   0.8958   (8 에서 포화)
#
# **교훈: 검색 knob 스윕은 그래프 상태에 종속적이다.** 연결이 성길 때 확장어는
# 잡음을 나르고, 연결이 촘촘해지면 신호를 나른다. 연결을 고치기 전의 튜닝은 그
# 시점 결함의 그림자를 쫓는 것이다. 그래서 8 을 "측정된 최적"으로 주장하지 않고
# **근거가 무효화된 변경을 철회**한 것으로 둔다.
#
# 6 이 2 와 8 보다 나쁜 것(비단조)도 그냥 적어 둔다 — 16 케이스에서 1건은
# 0.0625 이고, 이 요동은 튜닝 가능한 신호가 아니라 잡음이다. hit@5 는 이미
# 1.0 으로 포화했으므로 남은 신호는 hit@1 뿐이다. 더 큰 골든셋 없이 이 값을
# 더 만지는 것은 과적합이다.
DEFAULT_MAX_TERMS = 8

# 확장에 쓰는 계층 술어. is_a 는 온톨로지의 뼈대이고 KG 엔진이 전이 폐포를
# 이미 제공한다 (get_ancestors/get_descendants).
#
# ⚠️ 실측: ins_cancer_demo 에는 is_a 엣지가 **0개**다(술어는 definesTerm ·
# hasParty · coversDisease · hasCondition). 이 데이터에서 폐포 확장은 죽은
# 코드이고, 1-hop 인접도 노드의 62.9%가 고립이라 닿지 않는다. 그래서 확산
# 채널(graph_propagation)을 더했다 — 청크를 허브로 보면 고립이 3.6%로 떨어진다.
HIERARCHY_PREDICATE = "is_a"

# 동의어 술어 — 표기가 완전히 다른 동의어를 원문 근거로 잇는다.
#
# **왜 필요했나(실측)**: 관계 백필의 is_a 제안 11건 중 **3건이 동일시를 is_a 로
# 왜곡**했다. 원문은 명확히 같다고 말한다 — "‘계약 전 알릴 의무’라 하며, 상법상
# ‘고지의무’와 **같습니다**". 허용 어휘에 sameAs 가 없어서 LLM 이 가장 가까운
# is_a 로 밀어 넣은 것이다. 어휘 부족이 만든 체계적 오류이고 술어 하나로 고쳐진다.
#
# **병합이 아니라 술어인 이유**: `고지의무`(상법)와 `계약 전 알릴 의무`(약관)를
# 합치면 **문서가 구별한 것을 지운다** — 두 법이 다른 이름을 쓴다는 사실이 사라진다.
# 병합은 묘비도 없이 되돌릴 수 없다. sameAs 는 비파괴적이고 검색에서는 별칭과 같은
# 일을 한다. 중복 병합은 여전히 별도 경로(merge_nodes)로 남는다.
#
# 그리고 이건 중복 검사가 못 잡는 것을 잡는다 — graph_health 의 중복 클러스터는
# 이름 유사도 기반이라 표기가 완전히 다른 동의어를 보지 못한다(실측: 클러스터
# 4개에 이 쌍이 없다).
#
# **대칭으로 확장한다**(아래 expand 의 in_edges 처리). 동의어가 한 방향으로만
# 통하면 원문을 어느 쪽으로 적었는지라는 우연이 검색을 좌우한다.
# **전이는 하지 않는다** — A=B, B=C 를 A=C 로 넓히면 잘못된 sameAs 하나가 클러스터
# 전체를 오염시킨다. is_a 는 폐포를 쓰지만(계층은 틀려도 국소적) sameAs 는 1-hop 이다.
SYNONYM_PREDICATE = "sameAs"

# 확산 — 이분 그래프 Personalized PageRank (HippoRAG 2 / LinearRAG 계열).
#
# **두 절반은 성질이 다르므로 플래그도 둘이다.** 처음엔 노드 채점으로만 쟀는데
# (확장만 on = 동일 / 청크 채널만 on = hit@10 0.9375→0.8750) 그건 청크 채널을
# **노드 자로** 잰 숫자였다 — 청크 순서 변경이 retrieve_result_to_nodes 의 3번
# 소스를 재배열한 것. 그래서 청크 자(target="evidence", 새 라벨 0개로 성립)를
# 만들고 다시 쟀다. 16 케이스, k=5:
#
#   채널                      hit@1    hit@5    MRR
#   ──────────────────────────────────────────────────────────────
#   chunk (원 질의만)         0.5000   0.7500   0.5646   ← baseline
#   retrieve (확산 off)       0.5000   0.8125   0.6271   ← **그래프 조건화의 이득**
#   + 확장 절반만             0.5000   0.8125   0.6271   소수점까지 동일
#   + 청크 채널 (w=0.05)      0.5000   0.8125   0.6271   동일
#   + 청크 채널 (w=1.0)       0.4375   0.8125   0.5729   퇴화
#
# 읽어야 할 두 가지:
#  1) **그래프 조건화가 baseline 을 이긴다** — MRR 0.5646 → 0.6271. 노드 채점에서
#     semantic 과 retrieve 가 구조상 같게 나와 이 차이가 보이지 않았다. 청크
#     자리로 옮긴 첫 측정에서 축 4 의 가치가 숫자가 됐다.
#  2) 확산은 그 시점 데이터에서 이득 0 — hit@5 가 모든 가중치에서 0.8125 로 고정.
#
# ⚠️ **위 표는 고아 노드 27개를 안고 잰 값이다. 원인 진단이 두 번 뒤집혔다.**
#
# 처음엔 천장 0.8125(=13/16)를 "임베딩 리콜 실패"로 적었는데 틀렸다. 실패 3건의
# 정답 노드(특별부활 ×2 · 계약자적립액)는 임베더가 잘 찾는다 — 2위 0.763,
# **1위 0.729**. 진짜 원인은 그 노드들의 **근거 청크가 0개**라는 것이었다:
# 채널 B 가 끌어올 청크가 없고 evidence 채점은 정답 노드에 연결된 청크를 요구하니
# **구조적으로 불가능**했다. 확산도 링크가 없는 곳으로는 못 간다.
#
# 고아 23개에 근거 66개를 이은 뒤(approve_orphan_links) 같은 자로 재면:
#
#   채널                      hit@1    hit@5    MRR
#   ──────────────────────────────────────────────────
#   chunk (원 질의만)         0.6250   0.8750   0.6896
#   retrieve                  0.8125   1.0000   0.8958   ← 천장 돌파
#
# **천장은 검색이 아니라 연결이 정했다.** hit@5 가 16/16 이 됐고, 검색 knob 을
# 15개 조합 스윕해도 못 넘던 0.8125 가 링크 하나 없이 사라졌다. 그리고 확산의
# "이득 0" 도 연결 부재의 그림자였다 — 회복 후에는 w=0.25 에서 MRR 0.8958 로
# 기여한다(단, max_terms=8 과 동률 = 중복이므로 기본 off 유지).
#
# 그래서 둘 다 기본 off 를 **유지**한다. 이유가 "측정할 수 없어서"에서 "측정했고
# 이득이 0 이라서"로 바뀌었다. 확장 절반은 무해함이 두 자(노드·청크)에서 확인됐고,
# 청크 채널은 w>0.05 에서 단조 퇴화한다.
DEFAULT_USE_PROPAGATION = False        # 확장 절반 (노드 후보)
DEFAULT_PROPAGATION_CHANNEL = False    # 청크 채널 C
# 확산에서 가져올 상한. 질량 꼬리는 사실상 균등해서 순위 정보가 없다.
DEFAULT_PROPAGATION_TOP = 10

# 확산 채널의 RRF 가중 — **스윕으로 정한 값**(target="evidence", 16 케이스, k=5).
# 1.0 이면 확산이 청크·그래프 채널과 동등한 표를 갖고 확산 단독 청크가 정답 앞에
# 끼어든다:
#
#   w      hit@1    hit@5    MRR
#   ─────────────────────────────────
#   off    0.5000   0.8125   0.6271   ← 기준선(retrieve)
#   0.05   0.5000   0.8125   0.6271   ← 동률. 무해한 최대값
#   0.1    0.5000   0.8125   0.6219
#   0.25   0.5000   0.8125   0.6250
#   0.5    0.5000   0.8125   0.6250
#   0.75   0.5000   0.8125   0.6250
#   1.0    0.4375   0.8125   0.5729   ← 퇴화
#
# 0.05 는 HippoRAG 2 의 passage reset 가중(0.05)과 우연히 일치한다 — 문헌과 우리
# 스윕이 같은 감쇠 강도를 가리켰다. ⚠️ 16 케이스는 작다. "무해한 최대값"이지
# "최적값"이 아니다.
#
# ⚠️ 근거 링크 회복 후 재측정에서는 **0.25 가 더 좋았다** (MRR 0.8958 vs
# 0.8646). 그런데 그 값은 `max_terms=8`(확산 off)와 **정확히 동률**이다 —
# 확산과 확장어가 상보가 아니라 중복이라는 뜻이고, 그래서 확산을 켤 이유가
# 없어 0.05 를 그대로 둔다(확산은 기본 off). 켜려는 사람은 0.25 를 보라.
# 이 폭(0.05~0.25)의 차이는 16 케이스에서 1~2건이라 잡음과 구별되지 않는다.
DEFAULT_PROPAGATION_WEIGHT = 0.05


@dataclass
class Expansion:
    """질의 확장 결과 — 무엇이 왜 붙었는지 전부 드러난다."""
    query: str
    entry_nodes: List[Dict[str, Any]] = field(default_factory=list)
    expanded_nodes: List[Dict[str, Any]] = field(default_factory=list)
    terms: List[str] = field(default_factory=list)
    expanded_query: str = ""


@dataclass
class RetrievalHit:
    """검색 결과 한 건 — 항상 '왜 걸렸는지'를 들고 다닌다."""
    chunk: StoredChunk
    score: float
    matched_via: List[str] = field(default_factory=list)
    channels: List[str] = field(default_factory=list)
    # 동점 해소용. RRF 는 순위만 쓰므로 채널이 둘뿐일 때 1·2위가 맞바뀌면
    # 수학적으로 반드시 동점이 난다 (1/61+1/62 == 1/62+1/61). 실측에서 실제로
    # 났고, 그때 순서가 dict 삽입 순서로 갈렸다 — 임의적이라 그 자체로 버그다.
    best_evidence: float = 0.0


class GraphConditionedRetriever:
    """온톨로지로 조건화된 청크 검색."""

    def __init__(self, namespace: str = "default", kg=None, store=None,
                 index=None):
        self.namespace = namespace
        self._kg = kg
        self._store = store
        self._index = index

    # ─── 지연 자원 ───────────────────────────────────────────────────

    @property
    def kg(self):
        if self._kg is None:
            from ..engines.knowledge_graph_clean import get_knowledge_graph_engine
            self._kg = get_knowledge_graph_engine(self.namespace)
        return self._kg

    @property
    def store(self):
        if self._store is None:
            from .chunk_store import get_chunk_store
            self._store = get_chunk_store(self.namespace)
        return self._store

    @property
    def index(self):
        """chunk 채널의 검색기.

        aicoach-backed 네임스페이스(graph_store.aicoach_source)는 CHUNK 채널을
        aicoach 의 라이브 ES 인덱스(aicoach-kb)로 붙인다 — 로컬 npy 스냅샷 대신
        살아있는 약관을 인용하게. 그래프/노드 채널(채널 B)은 그대로다. 비-aicoach
        네임스페이스는 기존 로컬 청크 인덱스를 그대로 쓴다.
        """
        if self._index is None:
            from .graph_store import aicoach_source
            if aicoach_source(self.namespace):
                from .aicoach_chunk_index import get_aicoach_chunk_index
                self._index = get_aicoach_chunk_index(self.namespace)
            else:
                from .chunk_index import get_chunk_index
                self._index = get_chunk_index(self.namespace)
        return self._index

    # ─── 확장 ────────────────────────────────────────────────────────

    @staticmethod
    def _terms_of(node_id: str, attrs: Dict[str, Any]) -> List[str]:
        """노드가 기여하는 확장 어휘 — 이름 + 별칭.

        별칭이 핵심이다: aicoach 의 _SYN 하드코딩 테이블(은퇴↔연금/노후/퇴직)이
        하려던 일을 **데이터로** 한다. 새 동의어는 코드 수정이 아니라 온톨로지
        갱신으로 들어온다.
        """
        terms: List[str] = []
        name = attrs.get("name")
        if name:
            terms.append(str(name))
        aliases = attrs.get("aliases")
        if isinstance(aliases, (list, tuple)):
            terms.extend(str(a) for a in aliases if a)
        return terms

    def _propagated_entities(self, entry_nodes: List[Dict[str, Any]],
                             top: int) -> List[tuple]:
        """확산으로 닿은 **개체** 상위 — 죽은 1-hop 확장을 대체하는 절반.

        HippoRAG 2 는 PPR 로 phrase 노드와 passage 노드를 함께 순위 낸다.
        청크(passage) 절반만 쓰면 확산이 노드 랭킹·확장어에 기여하지 못한다.

        진입 노드 자신은 제외한다 — 이미 entry 로 들어가 있다.
        """
        seeds = {e["node_id"]: e.get("score", 0.0) for e in entry_nodes}
        if not seeds:
            return []
        try:
            from .graph_propagation import (background_masses, build_bipartite,
                                            lift, propagate, split_masses)
            bipartite = build_bipartite(self.kg.graph, self.store.all())
            entities, _ = split_masses(propagate(bipartite, seeds))
            # 배경 대비 배율로 정렬한다. 원 질량으로 정렬하면 상위가 전부
            # 허브였다 — "중증 갑상선암이란?" 의 1위가 `보험계약` 이었다
            # (PageRank 차수 편향). graph_propagation.lift 참고.
            # 배경 계산이 실패하면 확산을 **생략한다**. 원 질량으로 되돌아가면
            # 조용히 허브 편향 순위가 되는데, 그건 잡음을 확장어로 넣는 것이다.
            scored = lift(entities, background_masses(bipartite))
        except Exception as e:
            logger.warning(f"⚠️ Propagation expand failed ({e}) — 인접 확장만")
            return []
        ranked = [(nid, value) for nid, value in scored.items()
                  if nid not in seeds and value > 0.0
                  and entities.get(nid, 0.0) > 0.0]
        # 배율 내림차순, 동배율은 id 순 — 결정론성(부동소수 합산 순서에 흔들리면
        # 확장 집합이 질의마다 달라진다)
        ranked.sort(key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top]

    def expand(self, query: str,
               entry_k: int = DEFAULT_ENTRY_K,
               min_entry_score: float = DEFAULT_MIN_ENTRY_SCORE,
               entry_ratio: float = DEFAULT_ENTRY_RATIO,
               max_terms: int = DEFAULT_MAX_TERMS,
               use_propagation: bool = DEFAULT_USE_PROPAGATION,
               propagation_top: int = DEFAULT_PROPAGATION_TOP) -> Expansion:
        """질의 → 진입 노드 → 온톨로지 확장 → 확장 질의.

        진입은 의미 검색(임베딩)으로 찾고, 확장은 그래프 구조(is_a 폐포 +
        인접)로 넓힌다. 임베딩이 '대략 어디쯤'을 알려주면 그래프가 '정확히
        무엇'으로 데려가는 구조다.
        """
        expansion = Expansion(query=query, expanded_query=query)
        if not query or not query.strip():
            return expansion

        graph = self.kg.graph
        try:
            entries = self.kg.semantic_search(query, top_k=entry_k)
        except Exception as e:
            logger.warning(f"⚠️ Graph entry search failed ({e}) — chunk channel only")
            return expansion

        seen: set = set()
        terms: List[str] = []

        def add_terms(node_id: str, attrs: Dict[str, Any]) -> None:
            for term in self._terms_of(node_id, attrs):
                if term not in terms and term.lower() not in query.lower():
                    terms.append(term)

        # 1) 절대 바닥 — 유사도가 0 이하면 무관하다 (코사인의 정의)
        candidates = [e for e in entries
                      if e.get("score", 0.0) > min_entry_score
                      and e["node_id"] in graph]
        if not candidates:
            return expansion

        # 2) 상대 게이트 — 최고점 대비 비율. 척도-무관이라 임베더가 바뀌어도 성립.
        top_score = max(e["score"] for e in candidates)
        for entry in candidates:
            if entry["score"] < top_score * entry_ratio:
                continue
            node_id = entry["node_id"]
            expansion.entry_nodes.append(entry)
            seen.add(node_id)
            add_terms(node_id, graph.nodes[node_id])

        # 그래프 확장: is_a 상위/하위 폐포 + 직접 인접
        for entry in list(expansion.entry_nodes):
            node_id = entry["node_id"]
            related: List[tuple] = []
            try:
                for ancestor in self.kg.get_ancestors(node_id,
                                                      predicate=HIERARCHY_PREDICATE):
                    related.append((ancestor, "is_a↑"))
                for descendant in self.kg.get_descendants(node_id,
                                                          predicate=HIERARCHY_PREDICATE):
                    related.append((descendant, "is_a↓"))
            except Exception:
                pass  # 계층이 없는 그래프도 있다 — 인접만으로 계속한다
            for _, target, attrs in graph.out_edges(node_id, data=True):
                if attrs.get("predicate") != HIERARCHY_PREDICATE:
                    related.append((target, attrs.get("predicate", "")))
            # 동의어는 **대칭**이므로 역방향도 본다. 다른 술어는 out 방향만 —
            # 전부 대칭으로 만들면 확장이 무의미하게 폭발한다
            # (`hasParty` 의 역방향은 "이 당사자를 가진 모든 계약"이다).
            for source, _, attrs in graph.in_edges(node_id, data=True):
                if attrs.get("predicate") == SYNONYM_PREDICATE:
                    related.append((source, f"{SYNONYM_PREDICATE}↔"))

            for target, via in related:
                if target in seen or target not in graph:
                    continue
                seen.add(target)
                expansion.expanded_nodes.append(
                    {"node_id": target, "via": via, "from": node_id,
                     "name": graph.nodes[target].get("name", target)})
                add_terms(target, graph.nodes[target])

        # 확산 확장 — 개체 절반. 1-hop 이 닿지 못한 곳(실측 고립 62.9%)을 청크를
        # 허브로 건넌다. 인접 확장을 지우지 않고 뒤에 붙인다: 온톨로지가 명시한
        # 관계가 통계적 확산보다 앞이어야 한다.
        if use_propagation:
            for node_id, node_lift in self._propagated_entities(
                    expansion.entry_nodes, top=propagation_top):
                if node_id in seen or node_id not in graph:
                    continue
                seen.add(node_id)
                expansion.expanded_nodes.append(
                    {"node_id": node_id, "via": "propagation",
                     # 배경 대비 배율. 질량이 아니라 배율이다 — 이름을 mass 로
                     # 두면 다음 사람이 확률처럼 더하려 한다.
                     "from": "", "lift": round(node_lift, 4),
                     "name": graph.nodes[node_id].get("name", node_id)})
                # ⚠️ add_terms 를 **부르지 않는다**. 실측: 확산 노드의 이름을
                # 확장어에 넣으면 질의가 희석돼 청크 채널 결과가 바뀌고,
                # hit@10 이 0.9375 → 0.8750 로 떨어졌다(정답이 꼬리 밖으로).
                # 확장어는 원 질의에 직접 섞이는 자리라 정밀도가 가장 중요하고,
                # 온톨로지가 **명시한** 관계(1-hop)는 그 자격이 있지만 통계적
                # 확산은 없다 — 이 파일의 "확장어가 늘면 희석된다" 규정 그대로.
                # 확산은 노드 후보로만 기여한다(꼬리에 추가 → 순수 더하기).

        expansion.terms = terms[:max_terms]
        if expansion.terms:
            expansion.expanded_query = f"{query} {' '.join(expansion.terms)}"
        return expansion

    # ─── 검색 ────────────────────────────────────────────────────────

    def propagate_chunks(self, expansion: Expansion,
                         top: int = DEFAULT_PROPAGATION_TOP) -> List[tuple]:
        """확산 채널 — 이분 그래프 PPR 로 (청크, 질량) 순위를 낸다.

        채널 B(진입 노드에 달린 청크)와 다른 것을 본다: B 는 1-hop 이고, 여기는
        청크를 허브로 여러 홉을 건넌다. 개체-개체 엣지가 없어도 같은 조문에
        근거가 있으면 서로에게 닿는다 — 실측 그래프(고립 62.9%, is_a 0개)에서
        B 가 닿지 못하는 관계가 대부분이다.

        **순위는 lift(배경 대비 배율)다 — raw 질량이 아니다.** 노드 절반은
        처음부터 lift 를 썼는데 여기만 raw 로 정렬하고 있었고, 그 비대칭의 대가가
        evidence 채점(16 케이스, k=5)에서 드러났다:

            정렬          hit@5    MRR      by_tag 붕괴
            ────────────────────────────────────────────────────────
            raw mass      0.7500   0.5573   graph 0.33 · procedure 0.25
            lift          0.8125   0.5729   graph 0.67 · procedure 0.50

        여러 홉을 건너야 하는 절차 질의에서 허브 청크(총칙 조문 — 계약·보험료·해지가
        모두 달린)가 정답 청크를 밀어냈다. 청크는 차수가 곧 "몇 개체를 담았나"라서
        긴 총칙 조문이 구조적으로 유리하다.

        lift 는 태그 붕괴를 고쳤지만 hit@1(0.4375 vs 0.5000)은 남았다 — 그건
        정렬이 아니라 **융합 가중**의 문제였다(DEFAULT_PROPAGATION_WEIGHT 스윕 표).

        실패는 빈 목록 (never raise) — 확산은 보조 채널이다.
        """
        seeds = {e["node_id"]: e.get("score", 0.0)
                 for e in expansion.entry_nodes}
        if not seeds:
            return []
        try:
            from .graph_propagation import (background_masses, build_bipartite,
                                            lift, propagate, split_masses)
            bipartite = build_bipartite(self.kg.graph, self.store.all())
            _, chunk_masses = split_masses(propagate(bipartite, seeds))
            # 배경 계산이 실패하면 **생략한다**. raw 로 되돌아가면 조용히 허브
            # 편향 순위가 되는데, 그게 위 표의 퇴화다 (노드 절반과 같은 규정).
            _, chunk_background = split_masses(background_masses(bipartite))
            scored = lift(chunk_masses, chunk_background)
        except Exception as e:
            logger.warning(f"⚠️ Propagation channel failed ({e}) — 생략")
            return []

        ranked: List[tuple] = []
        for chunk_id, mass in sorted(scored.items(),
                                     key=lambda kv: (-kv[1], kv[0])):
            stored = self.store.get(chunk_id)
            if stored is None:
                continue        # 스토어에서 지워진 청크 — 허공을 가리킨다
            ranked.append((stored, mass))
            if len(ranked) >= top:
                break
        return ranked

    def search(self, query: str, top_k: int = 5,
               entry_k: int = DEFAULT_ENTRY_K,
               min_entry_score: float = DEFAULT_MIN_ENTRY_SCORE,
               entry_ratio: float = DEFAULT_ENTRY_RATIO,
               max_terms: int = DEFAULT_MAX_TERMS,
               use_propagation: bool = DEFAULT_USE_PROPAGATION,
               propagation_channel: bool = DEFAULT_PROPAGATION_CHANNEL,
               propagation_top: int = DEFAULT_PROPAGATION_TOP,
               propagation_weight: float = DEFAULT_PROPAGATION_WEIGHT,
               source: Optional[str] = None
               ) -> List[RetrievalHit]:
        """채널들을 RRF 로 융합한 청크 검색.

        확산은 두 갈래로 따로 켠다 (측정 결과가 다르다 — 모듈 상단 표 참고):
        `use_propagation` 은 확장(노드 후보), `propagation_channel` 은 청크
        채널 C 다.

        `source` 는 문서 필터("이 문서에서만"). **근거(청크)만 거르고 온톨로지
        (노드 확장)는 거르지 않는다** — 그래프는 네임스페이스 전체의 지식이고,
        문서 필터는 증거의 출처 제한이지 지식의 제한이 아니다. 다른 문서에서
        배운 별칭·계층이 이 문서의 청크를 찾는 데 쓰이는 것이 요점이다.
        **세 채널 전부** 거른다 — 하나라도 새면 "대체로 그 문서"가 되어 사용자가
        남의 문서 청크를 이 문서 것으로 오독한다.
        """
        if not query or not query.strip():
            return []
        if not len(self.store):
            # aicoach-backed 는 청크의 진실이 aicoach-kb(채널 A)라 로컬 스냅샷이
            # 비어도 검색이 성립한다 — 로컬 스토어 유무로 막지 않는다. 비-aicoach
            # 는 로컬 스토어가 유일한 청크 원천이므로 기존대로 단락한다.
            from .graph_store import aicoach_source
            if not aicoach_source(self.namespace):
                return []

        expansion = self.expand(query, entry_k=entry_k,
                                min_entry_score=min_entry_score,
                                entry_ratio=entry_ratio,
                                max_terms=max_terms,
                                use_propagation=use_propagation,
                                propagation_top=propagation_top)

        # 채널 A — 청크 검색. 확장 질의를 쓴다: 확장어가 있으면 원 질의만으로는
        # 글자가 안 겹쳐 놓치던 청크가 걸린다.
        # 상위 몇 개만 보면 그래프가 끌어올릴 청크가 애초에 후보에 없을 수
        # 있으므로 넉넉히 가져와 순위만 쓴다.
        # source 미지정이면 인자를 아예 넘기지 않는다 — 인덱스 구현이 여럿이고
        # (ChunkIndex · aicoach · 테스트 대역) 전부가 source 를 알 필요는 없다.
        # 지정됐는데 인덱스가 모르면(TypeError) **사후 필터로 degrade** 한다:
        # 굶주림(starvation)은 감수해도 필터 누수는 안 된다 — 남의 문서 청크가
        # 섞이면 사용자가 이 문서 것으로 오독한다.
        pool_k = max(top_k * 3, 10)
        if source is None:
            chunk_hits = self.index.search(expansion.expanded_query,
                                           top_k=pool_k)
        else:
            try:
                chunk_hits = self.index.search(expansion.expanded_query,
                                               top_k=pool_k, source=source)
            except TypeError:
                chunk_hits = [(c, s) for c, s in self.index.search(
                                  expansion.expanded_query, top_k=pool_k)
                              if getattr(c, "source", None) == source]

        # 채널 B — 그래프 경유. 진입/확장 노드에 달린 청크.
        graph_ranked: List[tuple] = []
        node_scores = {e["node_id"]: e.get("score", 0.0)
                       for e in expansion.entry_nodes}
        related_ids = [e["node_id"] for e in expansion.entry_nodes]
        related_ids += [n["node_id"] for n in expansion.expanded_nodes]
        for node_id in related_ids:
            for stored in self.store.chunks_for_node(node_id):
                if source is not None and stored.source != source:
                    continue      # 문서 필터 — 채널 B 도 같은 계약
                graph_ranked.append((stored, node_id,
                                     node_scores.get(node_id, 0.0)))
        # 진입 노드 점수 순 — 확장 노드(점수 0)는 뒤로 간다. 직접 걸린 개념이
        # 파생 개념보다 우선이어야 한다.
        graph_ranked.sort(key=lambda x: x[2], reverse=True)

        # ─── RRF 융합 ────────────────────────────────────────────────
        fused: Dict[str, RetrievalHit] = {}

        def contribute(stored: StoredChunk, rank: int, channel: str,
                       evidence: float = 0.0, via: Optional[str] = None,
                       score_rank: bool = True, weight: float = 1.0) -> None:
            hit = fused.get(stored.chunk_id)
            if hit is None:
                hit = RetrievalHit(chunk=stored, score=0.0)
                fused[stored.chunk_id] = hit
            if score_rank:
                # weight 는 채널의 표 크기다. 0 이면 "찾았다고 기록하되 순위에
                # 표는 주지 않는다" — provenance 는 남는다.
                hit.score += weight / (RRF_K + rank)
            hit.best_evidence = max(hit.best_evidence, evidence)
            if channel not in hit.channels:
                hit.channels.append(channel)
            if via and via not in hit.matched_via:
                hit.matched_via.append(via)

        for rank, (stored, score) in enumerate(chunk_hits, start=1):
            contribute(stored, rank, "chunk", evidence=score)

        seen_chunk_ids: set = set()
        rank = 0
        for stored, node_id, node_score in graph_ranked:
            if stored.chunk_id in seen_chunk_ids:
                # 같은 청크가 여러 노드로 걸리면 순위는 한 번만 매기고
                # 근거(matched_via)만 덧붙인다 — 노드가 많이 달렸다는 이유로
                # 점수가 부풀면 안 된다.
                contribute(stored, rank, "graph", evidence=node_score,
                           via=node_id, score_rank=False)
                continue
            rank += 1
            seen_chunk_ids.add(stored.chunk_id)
            contribute(stored, rank, "graph", evidence=node_score, via=node_id)

        # 채널 C — 확산(PPR)으로 찾은 청크. opt-in. ⚠️ 청크 자로 측정한 결과
        # **이득 0**이다(hit@5 가 모든 가중치에서 0.8125 — 새로 찾아주는 정답
        # 청크가 없다). 가중을 낮게 유지하는 것이 유일한 안전 조건이다.
        # 모듈 상단 표 + DEFAULT_PROPAGATION_WEIGHT 참고.
        if propagation_channel:
            # 문서 필터는 순위 매기기 **전에** 건다 — 거른 뒤 rank 를 매겨야
            # 걸러진 항목이 만든 순위 공백이 남은 항목의 RRF 기여를 깎지 않는다.
            propagated = [(stored, mass) for stored, mass
                          in self.propagate_chunks(expansion,
                                                   top=propagation_top)
                          if source is None or stored.source == source]
            for rank, (stored, mass) in enumerate(propagated, start=1):
                # via 에 확산임을 남긴다 — 어느 채널이 끌어올렸는지 보이지
                # 않으면 순위를 설명할 수 없다(이 파일의 계약).
                #
                # evidence 에 질량을 넣지 않는다. PPR 질량은 전체 합이 1 인
                # 분포값이라 코사인과 **비교 불가능한 척도**다. best_evidence 는
                # 동점을 가르는 비교에 쓰이므로, 거기에 다른 척도를 섞으면 이
                # 파일이 RRF 를 쓰는 이유(선형 결합 불가)를 스스로 위반한다.
                # 질량은 이 채널 안의 순위를 정하는 데 이미 다 쓰였다.
                contribute(stored, rank, "propagation", via="propagation",
                           weight=propagation_weight)

        # 2차 정렬 키 = best_evidence. RRF 동점일 때 삽입 순서로 갈리는 것은
        # 임의적이므로, 어떤 채널에서든 가장 강했던 원 증거로 가른다.
        # **이것은 측정된 규칙이 아니라 결정론성을 위한 최소 장치다** — 동점은
        # 애초에 두 채널이 서로 반대로 말하고 있다는 뜻이고, 그런 질의는 골든셋
        # 없이 옳게 가를 방법이 없다.
        results = sorted(fused.values(),
                         key=lambda h: (h.score, h.best_evidence), reverse=True)
        for hit in results:
            hit.score = round(hit.score, 6)
            hit.best_evidence = round(hit.best_evidence, 6)
        return results[:top_k]
