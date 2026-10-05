"""
LLM provider seam — 커널이 "어떤 LLM인가"에 의존하는 단 하나의 지점.

축 1(패키지 분리)의 핵심. 이전에는 builder/pipeline.py 와 core/llm_manager.py
가 각자 `logosai.utils.llm_client` 를 직접 import 했다. 그 결과 문서에서
개념 하나 뽑으려는 소비자가 에이전트 프레임워크 전체를 설치해야 했고,
실제로 aicoach 는 그 비용 때문에 KG 엔진을 재구현했다
(aicoach backend/app/kg/graph.py:1-10 "without pulling its heavy transitive deps").

해석 순서 (첫 성공 채택):
  1. 주입된 팩토리 (register_llm_factory) — 테스트·커스텀 경로. 항상 최우선.
  2. google-genai 네이티브 (Gemini) — 기본 경로. 의존성 1개.
  3. logosai LLMClient — 설치돼 있으면 사용 (기존 동작 보존, openai/anthropic).

없으면 LLMUnavailableError. **조용한 실패 금지** — 임베더(semantic_index)는
없으면 빈 결과로 degrade 하지만 LLM 은 다르다. 임베딩이 없으면 검색 품질이
떨어질 뿐이지만, LLM 이 없으면 추출이 통째로 비어 사용자가 빈 그래프를
"빌드 성공"으로 오해한다. 그래서 여기서는 던진다.
"""

import asyncio
import inspect
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

from loguru import logger

PROVIDERS = ("google", "openai", "anthropic", "openai_compatible")

DEFAULT_PROVIDER = "google"

# 빌더 전용 추출 모델 — 전역 llm_config(플래너용)와 독립.
# (기존 builder/pipeline.py:DEFAULT_EXTRACTION_MODEL 에서 이관)
DEFAULT_EXTRACTION_MODEL = os.environ.get(
    "ONTOLOGY_EXTRACTION_MODEL", "gemini-3.5-flash")

DEFAULT_MODELS = {
    "google": DEFAULT_EXTRACTION_MODEL,
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "openai_compatible": None,  # 로컬/오픈소스 — 모델명은 추측 불가, 명시 필수
}


class LLMUnavailableError(RuntimeError):
    """사용 가능한 LLM 경로가 하나도 없다."""


@dataclass
class ProviderSpec:
    """프로바이더 해석에 필요한 전부. 팩토리에 그대로 전달된다."""
    provider: str = DEFAULT_PROVIDER
    model: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    base_url: Optional[str] = None


class LLMProvider(Protocol):
    """커널이 LLM 에 요구하는 전부: 프롬프트 하나 → 텍스트 하나."""

    async def complete(self, prompt: str) -> str: ...


# ─── 구현체들 ────────────────────────────────────────────────────────

class CallableProvider:
    """sync/async 함수를 프로바이더로 감싼다 (주입·테스트 경로).

    sync 함수는 to_thread 로 돌린다 — 블로킹 SDK 호출이 이벤트 루프를
    멈추면 빌드 진행률 스트리밍이 함께 멈춘다.
    """

    def __init__(self, fn: Callable[[str], Any]):
        self._fn = fn

    async def complete(self, prompt: str) -> str:
        if inspect.iscoroutinefunction(self._fn):
            result = await self._fn(prompt)
        else:
            result = await asyncio.to_thread(self._fn, prompt)
        return result if isinstance(result, str) else str(result)


class GeminiProvider:
    """google-genai 네이티브 Gemini 클라이언트.

    logosai 없이 동작하는 기본 경로. 기존 core/llm_manager.GeminiLLMWrapper
    와 같은 SDK 호출 형태를 쓰되, langchain 스타일 메시지 객체를 요구하지
    않는다 (커널은 프롬프트 문자열만 안다).
    """

    def __init__(self, model: str, temperature: float = 0.1,
                 max_tokens: Optional[int] = None, api_key: Optional[str] = None):
        from google import genai

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise LLMUnavailableError(
                "GOOGLE_API_KEY is not set — Gemini provider needs it")
        self._client = genai.Client(api_key=api_key)

    def _generate(self, prompt: str) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(temperature=self.temperature)
        if self.max_tokens:
            config.max_output_tokens = self.max_tokens
        response = self._client.models.generate_content(
            model=self.model, config=config, contents=prompt)
        return response.text or ""

    async def complete(self, prompt: str) -> str:
        # SDK 호출은 블로킹 — 이벤트 루프를 비워둔다
        return await asyncio.to_thread(self._generate, prompt)


class LogosAIProvider:
    """logosai LLMClient 래퍼 — 설치돼 있을 때만 쓰이는 선택 경로.

    openai / anthropic / openai_compatible 은 아직 이 경로로만 간다.
    google 은 GeminiProvider 가 우선이고, google-genai 가 없을 때만 여기로
    떨어진다.
    """

    def __init__(self, provider: str, model: str, temperature: float = 0.1,
                 max_tokens: Optional[int] = None, base_url: Optional[str] = None):
        from logosai.utils.llm_client import LLMClient

        kwargs = {"provider": provider, "model": model,
                  "temperature": temperature, "max_tokens": max_tokens}
        if provider == "openai_compatible":
            kwargs["provider"] = "openai"
            kwargs["base_url"] = base_url
            kwargs["api_key"] = os.environ.get("OPENAI_COMPAT_API_KEY")
        self._client = LLMClient(**kwargs)

    async def complete(self, prompt: str) -> str:
        from logosai.utils.llm_client import GoogleLangChainWrapper

        wrapper = GoogleLangChainWrapper(self._client)
        response = await wrapper.ainvoke(prompt)
        content = getattr(response, "content", response)
        return content if isinstance(content, str) else str(content)


# ─── 가용성 프로브 (테스트가 monkeypatch 하는 지점) ──────────────────

def _gemini_available() -> bool:
    try:
        import google.genai  # noqa: F401
        return True
    except ImportError:
        return False


def _logosai_available() -> bool:
    try:
        import logosai.utils.llm_client  # noqa: F401
        return True
    except ImportError:
        return False


# ─── 팩토리 레지스트리 ───────────────────────────────────────────────

_factory: Optional[Callable[[ProviderSpec], LLMProvider]] = None


def register_llm_factory(factory: Optional[Callable[[ProviderSpec], LLMProvider]]) -> None:
    """전역 LLM 팩토리를 주입한다. 해석 순서에서 항상 최우선.

    소비 프로젝트가 자기 LLM 스택(사내 게이트웨이, vLLM 등)을 그대로 쓰게
    하는 seam. 테스트는 결정론적 가짜를 여기에 꽂는다.
    """
    global _factory
    _factory = factory


def reset_llm_factory() -> None:
    """주입을 해제하고 기본 해석 순서로 되돌린다."""
    global _factory
    _factory = None


# ─── 해석 ────────────────────────────────────────────────────────────

def _validate(spec: ProviderSpec) -> None:
    if spec.provider not in PROVIDERS:
        raise ValueError(
            f"unknown llm_provider '{spec.provider}' "
            f"(available: {', '.join(PROVIDERS)})")
    if not spec.model:
        raise ValueError(f"llm_model is required for provider '{spec.provider}'")
    if spec.provider == "openai_compatible" and not spec.base_url:
        raise ValueError("openai_compatible provider requires llm_base_url")


def resolve_provider(provider: str = DEFAULT_PROVIDER,
                     model: Optional[str] = None,
                     temperature: float = 0.1,
                     max_tokens: Optional[int] = None,
                     base_url: Optional[str] = None) -> LLMProvider:
    """프로바이더 인스턴스를 해석한다. 커널의 단일 진입점.

    provider 가 미지원이면 ValueError, 지원하지만 쓸 수 있는 백엔드가
    없으면 LLMUnavailableError.
    """
    if provider not in PROVIDERS:
        raise ValueError(
            f"unknown llm_provider '{provider}' "
            f"(available: {', '.join(PROVIDERS)})")

    spec = ProviderSpec(provider=provider,
                        model=model or DEFAULT_MODELS.get(provider),
                        temperature=temperature, max_tokens=max_tokens,
                        base_url=base_url)
    _validate(spec)

    if _factory is not None:
        return _factory(spec)

    if spec.provider == "google":
        if _gemini_available():
            return GeminiProvider(spec.model, spec.temperature,
                                  spec.max_tokens)
        if _logosai_available():
            logger.info("🤖 google-genai unavailable — falling back to logosai LLMClient")
            return LogosAIProvider("google", spec.model, spec.temperature,
                                   spec.max_tokens)
        raise LLMUnavailableError(
            "no LLM backend available for provider 'google' — "
            "install google-genai (pip install google-genai) or logosai")

    if _logosai_available():
        return LogosAIProvider(spec.provider, spec.model, spec.temperature,
                               spec.max_tokens, spec.base_url)
    raise LLMUnavailableError(
        f"no LLM backend available for provider '{spec.provider}' — "
        f"install logosai, or use provider='google' with google-genai")
