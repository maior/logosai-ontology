"""generic_grounded_agent — 도메인-제네릭 grounded 검색·템플릿 에이전트 (로드맵 5).

**네임스페이스마다 에이전트를 새로 만들지 않는다** — 이 클래스 하나에 agents.json
config row 가 도메인을 정한다 (docs/generic-grounded-agent-design.md P-1):

  mode=qa       : /retrieve 근거로 인용 답변 (grounded_qa 의 제네릭판 + source 한정)
  mode=template : 슬롯별로 근거를 **따로** 회수해 절을 조립 → 최종 문서(마크다운)
                  — PROJ-A 의 최종 산출물("문서 검색 → 가공 → 템플릿")이 이 모드다.

슬롯별 회수인 이유: 한 번의 회수로 문서 전체를 쓰면 임베딩이 강한 절이 top_k 를
독식해 나머지 절이 기아한다 — proposal_coverage 에서 실측한 그 결함(5/5 독식)의
템플릿판. 슬롯이 자기 질의·자기 source 필터·자기 top_k 를 가진다.

위치 무관 로드(메커니즘 B, 8915). grounding 은 HTTP — 커널 import 없음.
"""

import asyncio
import os
import sys
from typing import Any, Dict, List, Optional

from loguru import logger
from logosai import SimpleAgent, AgentResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grounded_common as gc  # noqa: E402
import generic_config as gcf  # noqa: E402

__all__ = ["GenericGroundedSearchAgent"]

DEFAULT_NAMESPACE = os.environ.get("ONTOLOGY_DEFAULT_NAMESPACE", "default")
ONTOLOGY_BASE_DEFAULT = "http://localhost:9274"
SLOT_LLM_TIMEOUT = 25


class GenericGroundedSearchAgent(SimpleAgent):
    """도메인 = config row. qa(인용 답변) / template(슬롯 조립 문서) 두 모드."""

    agent_name = "generic_grounded_search"
    agent_description = (
        "온톨로지 네임스페이스 근거 기반 제네릭 검색·문서산출 에이전트. "
        "agents.json parameters 로 도메인(namespace·모드·템플릿)을 받는다."
    )

    def __init__(self, config=None):
        super().__init__(config)
        # 로더는 row 의 parameters 를 AgentConfig.config 로 넘긴다.
        # 잘못된 row 여도 생성자는 던지지 않는다 — 던지면 로더가 None 을 돌려
        # 유령 에이전트가 된다. 오류는 보관했다가 handle 에서 소리낸다.
        params = getattr(config, "config", None) if config is not None else None
        self.gcfg, self.gcfg_errors = gcf.parse_agent_params(params)

    # ── 공통 부품 ────────────────────────────────────────────────────────
    def _text(self, query: Any, ctx: Dict[str, Any]) -> str:
        if isinstance(query, dict):
            return str(query.get("query") or query.get("text") or ctx.get("query") or "")
        return str(query or ctx.get("query") or "")

    async def _documents(self, namespace: str):
        return await self.call_tool_http(
            "GET", f"/api/v1/ontology/graphs/{namespace}/documents",
            base_url_env="ONTOLOGY_BASE", base_url_default=ONTOLOGY_BASE_DEFAULT,
        )

    async def _retrieve(self, namespace: str, query: str, top_k: int,
                        source: Optional[str] = None):
        params: Dict[str, Any] = {"query": query, "top_k": top_k}
        if source:
            params["source"] = source
        resp = await self.call_tool_http(
            "GET", f"/api/v1/ontology/graphs/{namespace}/retrieve",
            params=params,
            base_url_env="ONTOLOGY_BASE", base_url_default=ONTOLOGY_BASE_DEFAULT,
        )
        self._last_quality_note = gc.quality_note(resp)
        # 실패와 "근거 없음"을 구별하려면 오류를 여기서 붙잡아야 한다 —
        # parse_hits 는 둘 다 빈 리스트로 흡수한다(graceful relay).
        self._last_retrieval_error = gc.retrieval_error(resp)
        return gc.parse_hits(resp)

    async def _resolve_source(self, namespace: str, markers) -> Optional[str]:
        """마커 → 실문서 source (서버측 필터용). 마커 없으면 None(전체)."""
        if not markers:
            return None
        docs = await self._documents(namespace)
        return gc.pick_source(docs, tuple(markers))

    async def _compose(self, query: str, hits, instruction: str, fallback: str) -> str:
        prompt = gc.build_grounding_prompt(query, hits, instruction=instruction or None)
        try:
            resp = await asyncio.wait_for(self.llm_client.invoke(prompt),
                                          timeout=SLOT_LLM_TIMEOUT)
            text = resp.content if hasattr(resp, "content") else str(resp)
            if text and text.strip():
                return text.strip()
        except Exception as e:
            logger.warning(f"[{self.agent_name}] compose 실패 → 폴백: {e}")
        return fallback

    # ── mode=qa ──────────────────────────────────────────────────────────
    async def _handle_qa(self, namespace: str, text: str) -> AgentResponse:
        cfg = self.gcfg
        source = await self._resolve_source(namespace, cfg["source_markers"])
        hits = await self._retrieve(namespace, text, cfg["top_k"], source=source)
        if not hits:
            err = getattr(self, "_last_retrieval_error", None)
            msg = gc.no_evidence_message(namespace, err)
            return AgentResponse.success(
                content={"answer": msg, "citations": [], "namespace": namespace,
                         "retrieval_error": err},
                message=msg)
        answer = await self._compose(
            text, hits, cfg["instruction"],
            fallback="\n".join(f"- {str(h.get('text') or '').strip()[:120]} [{i}]"
                               for i, h in enumerate(hits, 1)))
        note = getattr(self, "_last_quality_note", "")
        if note:
            answer = f"{answer}\n\n_{note}_"
        cites = gc.citations_from_hits(hits)
        return AgentResponse.success(
            content={"answer": answer, "citations": cites, "namespace": namespace,
                     "source": source, "result": answer},
            message=answer)

    # ── mode=template ────────────────────────────────────────────────────
    async def _fill_slot(self, namespace: str, text: str, slot: Dict[str, Any]) -> Dict[str, Any]:
        markers = slot["source_markers"] or self.gcfg["source_markers"]
        source = await self._resolve_source(namespace, markers)
        sq = gcf.slot_query(text, slot)
        hits = await self._retrieve(namespace, sq, slot["top_k"], source=source)
        if not hits:
            # 부재면 자리를 지운다 — assemble 이 "근거를 찾지 못했습니다"로
            # 정직하게 채운다. **실패는 다르다**: 빈 자리로 두면 보고서가
            # "이 절엔 근거가 없다"로 읽혀 장애가 사실로 굳는다.
            err = getattr(self, "_last_retrieval_error", None)
            return {"slot_id": slot["slot_id"], "title": slot["title"],
                    "answer": gc.no_evidence_message(namespace, err) if err else "",
                    "citations": [], "source": source, "query": sq,
                    "retrieval_error": err}
        answer = await self._compose(
            sq, hits, slot["instruction"] or self.gcfg["instruction"],
            fallback="\n".join(f"- {str(h.get('text') or '').strip()[:120]} [{i}]"
                               for i, h in enumerate(hits, 1)))
        return {"slot_id": slot["slot_id"], "title": slot["title"], "answer": answer,
                "citations": gc.citations_from_hits(hits), "source": source, "query": sq}

    async def _handle_template(self, namespace: str, text: str) -> AgentResponse:
        tpl = self.gcfg["template"]
        # 슬롯은 병렬로 — 각 슬롯이 독립 회수+조립이라 순차일 이유가 없다.
        filled = list(await asyncio.gather(
            *(self._fill_slot(namespace, text, s) for s in tpl["slots"])))
        note = getattr(self, "_last_quality_note", "")
        answer = gcf.assemble_template(tpl["title"], filled, note=note)
        evidence_slots = sum(1 for s in filled if s["citations"])
        logger.info(f"[{self.agent_name}] template ns={namespace} "
                    f"slots={len(filled)} grounded={evidence_slots} q={text[:40]}")
        return AgentResponse.success(
            content={"answer": answer, "namespace": namespace, "result": answer,
                     "template": {"title": tpl["title"], "slots": filled},
                     "citations": [c for s in filled for c in s["citations"]]},
            message=answer)

    # ── 진입점 ───────────────────────────────────────────────────────────
    async def handle(self, query, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        ctx = context or {}
        if self.gcfg_errors:
            # 잘못된 config row 는 여기서 소리낸다 (조용한 기본값 동작 금지).
            msg = "에이전트 설정(parameters) 오류: " + " / ".join(self.gcfg_errors)
            return AgentResponse.error(message=msg, content={"answer": msg})
        namespace = ctx.get("namespace") or self.gcfg["namespace"] or DEFAULT_NAMESPACE
        text = self._text(query, ctx)
        if not text.strip():
            msg = "질의가 비어 있습니다."
            return AgentResponse.error(message=msg, content={"answer": msg})
        if self.gcfg["mode"] == "template":
            return await self._handle_template(namespace, text)
        return await self._handle_qa(namespace, text)


if __name__ == "__main__":
    async def _t():
        from logosai.config import AgentConfig
        from logosai.types import AgentType
        cfg = AgentConfig(name="t", agent_type=AgentType.CUSTOM, description="t", config={
            "namespace": "ins_cancer_demo", "mode": "qa",
            "instruction": "보험 약관 근거로 답하라.",
        })
        a = GenericGroundedSearchAgent(cfg)
        await a.initialize()
        r = await a.process("청약 철회 기간은?", {})
        print(r.content.get("answer") if hasattr(r, "content") else r)
    asyncio.run(_t())
