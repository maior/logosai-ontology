"""
축 1 — LLM 프로바이더 seam.

커널이 "어떤 LLM인가"에 의존하는 지점을 이 파일 하나로 좁힌다. 이전에는
builder/pipeline.py 와 core/llm_manager.py 가 각자 `logosai.utils.llm_client`
를 직접 import 했고, 그래서 문서에서 개념 하나 뽑으려는 소비자가 에이전트
프레임워크를 통째로 설치해야 했다.

고정하는 계약:
- 기본 프로바이더는 google(Gemini) — google-genai 네이티브 경로.
- logosai 없이도 Gemini 가 동작한다. 있으면 쓰고, 없으면 안 쓴다.
- 주입(register_llm_factory / CallableProvider)이 항상 최우선.
- 사용 가능한 경로가 없으면 LLMUnavailableError 를 던진다 — 조용한 실패 금지.
  (임베더는 degrade 하지만 LLM 은 다르다: 임베딩이 없으면 검색 품질이
   떨어질 뿐이나, LLM 이 없으면 추출 결과가 통째로 비어 사용자가 빈
   그래프를 "빌드 성공"으로 오해한다.)
"""

import asyncio

import pytest

from ontology.core.llm_provider import (
    DEFAULT_MODELS,
    DEFAULT_PROVIDER,
    PROVIDERS,
    CallableProvider,
    LLMUnavailableError,
    register_llm_factory,
    reset_llm_factory,
    resolve_provider,
)


@pytest.fixture(autouse=True)
def _clean_factory():
    """각 테스트는 전역 팩토리 오염 없이 시작/종료한다."""
    reset_llm_factory()
    yield
    reset_llm_factory()


def _run(coro):
    return asyncio.run(coro)


class TestDefaults:
    def test_default_provider_is_google(self):
        assert DEFAULT_PROVIDER == "google"

    def test_google_has_a_default_model(self):
        assert DEFAULT_MODELS["google"]

    def test_openai_compatible_has_no_default_model(self):
        # 로컬/오픈소스 서버는 모델명을 추측할 수 없다 — 명시 필수
        assert DEFAULT_MODELS["openai_compatible"] is None

    def test_known_providers(self):
        assert set(PROVIDERS) == {"google", "openai", "anthropic",
                                  "openai_compatible"}


class TestCallableProvider:
    """주입된 함수는 그대로 프로바이더가 된다 (테스트·커스텀 경로)."""

    def test_sync_callable(self):
        provider = CallableProvider(lambda prompt: f"echo:{prompt}")
        assert _run(provider.complete("hi")) == "echo:hi"

    def test_async_callable(self):
        async def fn(prompt):
            return f"async:{prompt}"

        provider = CallableProvider(fn)
        assert _run(provider.complete("hi")) == "async:hi"

    def test_non_string_return_is_coerced(self):
        # LLM 래퍼가 객체를 돌려주는 경우가 있다 — 호출부는 str 을 기대한다
        provider = CallableProvider(lambda prompt: 42)
        assert _run(provider.complete("hi")) == "42"


class TestFactoryOverride:
    """주입이 최우선 — 실제 SDK 유무와 무관하게 결정론적으로 테스트된다."""

    def test_registered_factory_wins(self):
        register_llm_factory(lambda spec: CallableProvider(
            lambda prompt: f"{spec.provider}/{spec.model}"))
        provider = resolve_provider("google", "gemini-x")
        assert _run(provider.complete("q")) == "google/gemini-x"

    def test_factory_receives_resolved_default_model(self):
        seen = {}

        def factory(spec):
            seen["model"] = spec.model
            return CallableProvider(lambda p: "ok")

        register_llm_factory(factory)
        resolve_provider("google")  # 모델 미지정
        assert seen["model"] == DEFAULT_MODELS["google"]

    def test_reset_restores_default_resolution(self):
        register_llm_factory(lambda spec: CallableProvider(lambda p: "fake"))
        reset_llm_factory()
        # 팩토리가 사라졌으므로 실제 해석 경로를 탄다 — 여기서는 미지원
        # 프로바이더로 확인 (SDK 유무에 의존하지 않는 검증)
        with pytest.raises(ValueError):
            resolve_provider("nope")


class TestValidation:
    def test_unknown_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown llm_provider"):
            resolve_provider("no_such_provider")

    def test_openai_compatible_requires_base_url(self):
        register_llm_factory(lambda spec: CallableProvider(lambda p: "ok"))
        with pytest.raises(ValueError, match="base_url"):
            resolve_provider("openai_compatible", "qwen2.5-7b")

    def test_openai_compatible_requires_model(self):
        register_llm_factory(lambda spec: CallableProvider(lambda p: "ok"))
        with pytest.raises(ValueError, match="model"):
            resolve_provider("openai_compatible", base_url="http://x")


class TestNoSilentFailure:
    """사용 가능한 경로가 없으면 반드시 던진다."""

    def test_raises_when_no_backend_available(self, monkeypatch):
        import ontology.core.llm_provider as mod

        monkeypatch.setattr(mod, "_gemini_available", lambda: False)
        monkeypatch.setattr(mod, "_logosai_available", lambda: False)
        with pytest.raises(LLMUnavailableError):
            resolve_provider("google")

    def test_error_names_the_install_hint(self, monkeypatch):
        import ontology.core.llm_provider as mod

        monkeypatch.setattr(mod, "_gemini_available", lambda: False)
        monkeypatch.setattr(mod, "_logosai_available", lambda: False)
        with pytest.raises(LLMUnavailableError, match="google-genai"):
            resolve_provider("google")
