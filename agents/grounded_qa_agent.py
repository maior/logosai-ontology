"""
grounded_qa_agent — 온톨로지 근거 기반 질의응답 (환각 0).

Logos/ontology/agents/ 에 사는 acp 에이전트. ACP 는 파일 경로로 로드하므로
위치 무관(메커니즘 B, 별도 인스턴스 포트 8915). grounding 은 온톨로지
`/retrieve`(HTTP) — 온톨로지 커널을 파이썬 import 하지 않는다(경계 안전).

흐름: query → /retrieve(namespace) → 근거 발췌만으로 LLM 조립 + 번호 인용.
namespace 는 context["namespace"](없으면 기본값). aicoach clause_evidence 가
약관을 인용하던 원리를 도메인-무관 검색으로 옮긴 것.
"""

import asyncio
import os
import sys
from typing import Any, Dict, Optional

from loguru import logger
from logosai import SimpleAgent, AgentResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grounded_common as gc  # noqa: E402

__all__ = ["GroundedQAAgent"]

DEFAULT_NAMESPACE = os.environ.get("ONTOLOGY_DEFAULT_NAMESPACE", "default")
ONTOLOGY_BASE_DEFAULT = "http://localhost:9274"
TOP_K = 6


class GroundedQAAgent(SimpleAgent):
    """온톨로지 근거 기반 질의응답 — /retrieve 발췌만으로 인용 답변(환각 0)."""

    agent_name = "grounded_qa_agent"
    agent_description = (
        "업로드된 문서 지식베이스(온톨로지 네임스페이스)에 근거해 질문에 답한다. "
        "온톨로지 /retrieve 로 관련 원문을 회수하고, 그 발췌만으로 답을 조립하며 "
        "각 문장에 근거 번호를 인용한다(환각 0). 근거가 없으면 없다고 답한다. "
        "namespace 는 context 로 주입(기본 default)."
    )

    def _text(self, query: Any, ctx: Dict[str, Any]) -> str:
        if isinstance(query, dict):
            return str(query.get("query") or query.get("text") or ctx.get("query") or "")
        return str(query or ctx.get("query") or "")

    async def _retrieve(self, namespace: str, query: str, top_k: int = TOP_K):
        resp = await self.call_tool_http(
            "GET", f"/api/v1/ontology/graphs/{namespace}/retrieve",
            params={"query": query, "top_k": top_k},
            base_url_env="ONTOLOGY_BASE", base_url_default=ONTOLOGY_BASE_DEFAULT,
        )
        self._last_quality_note = gc.quality_note(resp)
        # 실패와 "근거 없음"을 구별하려면 오류를 여기서 붙잡아야 한다 —
        # parse_hits 는 둘 다 빈 리스트로 흡수한다(graceful relay).
        self._last_retrieval_error = gc.retrieval_error(resp)
        return gc.parse_hits(resp)

    async def _compose(self, query: str, hits, instruction: Optional[str] = None) -> str:
        """근거 발췌만으로 LLM 조립. 실패 시 최상위 근거로 graceful degrade."""
        prompt = gc.build_grounding_prompt(query, hits, instruction=instruction)
        try:
            resp = await asyncio.wait_for(self.llm_client.invoke(prompt), timeout=20)
            text = resp.content if hasattr(resp, "content") else str(resp)
            if text and text.strip():
                return text.strip()
        except Exception as e:
            logger.warning(f"[{self.agent_name}] compose LLM 실패 → 근거 발췌 폴백: {e}")
        # 폴백: 최상위 근거 원문(여전히 사실만)
        top = hits[0]
        return f"{str(top.get('text') or '').strip()} [1]"

    async def handle(self, query, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        ctx = context or {}
        namespace = ctx.get("namespace") or DEFAULT_NAMESPACE
        text = self._text(query, ctx)
        if not text.strip():
            msg = "질의가 비어 있습니다."
            return AgentResponse.error(message=msg, content={"answer": msg})

        hits = await self._retrieve(namespace, text)
        if not hits:
            err = getattr(self, "_last_retrieval_error", None)
            msg = gc.no_evidence_message(namespace, err)
            return AgentResponse.success(
                content={"answer": msg, "citations": [], "namespace": namespace,
                         "retrieval_error": err}, message=msg)

        answer = await self._compose(text, hits)
        # 측정된 품질을 답 꼬리에 — 지어낸 신뢰도가 아니라 골든셋 측정치다.
        note = getattr(self, "_last_quality_note", "")
        if note:
            answer = f"{answer}\n\n_{note}_"
        cites = gc.citations_from_hits(hits)
        logger.info(f"[{self.agent_name}] ns={namespace} hits={len(hits)} q={text[:40]}")
        return AgentResponse.success(
            content={"answer": answer, "citations": cites, "namespace": namespace,
                     "result": answer},
            message=answer,
        )


if __name__ == "__main__":
    async def _t():
        a = GroundedQAAgent()
        await a.initialize()
        r = await a.process("데이터 표준화 요구사항이 뭐야?", {"namespace": "PROJ-A"})
        print(r.content.get("answer") if hasattr(r, "content") else r)
    asyncio.run(_t())
