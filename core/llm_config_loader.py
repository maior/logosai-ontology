"""
🔧 LLM 설정 로더
LLM Configuration Loader

YAML 설정 파일을 읽어서 LLM 설정을 관리하는 모듈
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from loguru import logger

from .models import OntologyLLMConfig
from .llm_manager import LLMProvider, OntologyLLMType


@dataclass
class ModelMapping:
    """모델 매핑 정보"""
    provider: str
    model_tier: str
    model_name: str
    reason: Optional[str] = None


class LLMConfigLoader:
    """LLM 설정 로더"""
    
    def __init__(self, config_path: Optional[str] = None):
        """초기화"""
        if config_path is None:
            # 기본 설정 파일 경로
            config_path = Path(__file__).parent.parent / "config" / "llm_config.yaml"
        
        self.config_path = Path(config_path)
        self.config_data: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self):
        """설정 파일 로드"""
        try:
            if not self.config_path.exists():
                logger.warning(f"설정 파일을 찾을 수 없습니다: {self.config_path}")
                self._create_default_config()
                return
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_data = yaml.safe_load(f)
            
            logger.info(f"✅ LLM 설정 파일 로드 완료: {self.config_path}")
            
        except Exception as e:
            logger.error(f"❌ 설정 파일 로드 실패: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """기본 설정 생성"""
        logger.info("기본 설정을 생성합니다...")
        self.config_data = {
            "providers": {
                "google": {
                    "api_key_env": "GOOGLE_API_KEY",
                    "models": {
                        "high_performance": "gemini-2.0-flash-lite",
                        "standard": "gemini-2.0-flash-lite",
                        "fast": "gemini-2.0-flash-lite",
                        "creative": "gemini-2.0-flash-lite",
                        "budget": "gemini-2.0-flash-lite"
                    }
                }
            },
            "profiles": {
                "default": {
                    "description": "기본 설정 (Gemini)",
                    "settings": {
                        "semantic_analyzer": {"provider": "google", "model_tier": "fast"},
                        "workflow_designer": {"provider": "google", "model_tier": "fast"},
                        "knowledge_reasoner": {"provider": "google", "model_tier": "fast"},
                        "result_integrator": {"provider": "google", "model_tier": "fast"},
                        "query_processor": {"provider": "google", "model_tier": "fast"},
                        "graph_builder": {"provider": "google", "model_tier": "fast"},
                        "performance_optimizer": {"provider": "google", "model_tier": "fast"},
                        "creative_reasoner": {"provider": "google", "model_tier": "fast"}
                    }
                }
            },
            "default_profile": "all_gemini"
        }
    
    def get_available_providers(self) -> List[str]:
        """사용 가능한 제공업체 목록"""
        return list(self.config_data.get("providers", {}).keys())
    
    def get_available_profiles(self) -> List[str]:
        """사용 가능한 프로파일 목록"""
        return list(self.config_data.get("profiles", {}).keys())
    
    def get_provider_models(self, provider: str) -> Dict[str, str]:
        """제공업체의 모델 목록"""
        providers = self.config_data.get("providers", {})
        if provider not in providers:
            return {}
        
        return providers[provider].get("models", {})
    
    def get_model_name(self, provider: str, model_tier: str) -> str:
        """모델 이름 조회"""
        models = self.get_provider_models(provider)
        return models.get(model_tier, models.get("standard", "gpt-4o-mini"))
    
    def get_profile_settings(self, profile_name: str) -> Dict[str, Any]:
        """프로파일 설정 조회"""
        profiles = self.config_data.get("profiles", {})
        if profile_name not in profiles:
            logger.warning(f"프로파일을 찾을 수 없습니다: {profile_name}")
            return {}
        
        return profiles[profile_name].get("settings", {})
    
    def get_ontology_llm_defaults(self, llm_type: str) -> Dict[str, Any]:
        """온톨로지 LLM 타입별 기본 설정"""
        defaults = self.config_data.get("ontology_llm_defaults", {})
        return defaults.get(llm_type, {})
    
    def create_llm_config(self, llm_type_str: str, provider: str, model_tier: str) -> OntologyLLMConfig:
        """LLM 설정 객체 생성"""
        try:
            # 제공업체 enum 변환
            provider_enum = LLMProvider(provider.lower())
            
            # 모델 이름 조회
            model_name = self.get_model_name(provider, model_tier)
            
            # 기본 설정 조회
            defaults = self.get_ontology_llm_defaults(llm_type_str)
            provider_defaults = self.config_data.get("providers", {}).get(provider, {}).get("default_params", {})
            
            # 설정 병합 (우선순위: ontology_llm_defaults > provider_defaults)
            config_params = {}
            config_params.update(provider_defaults)
            config_params.update(defaults)
            
            # OntologyLLMConfig 생성
            config = OntologyLLMConfig(
                provider=provider_enum,
                model=model_name,
                temperature=config_params.get("temperature", 0.7),
                max_tokens=config_params.get("max_tokens", 3000),
                top_p=config_params.get("top_p", 1.0),
                frequency_penalty=config_params.get("frequency_penalty", 0.0),
                presence_penalty=config_params.get("presence_penalty", 0.0),
                request_timeout=config_params.get("request_timeout"),
                max_retries=config_params.get("max_retries", 3),
                streaming=config_params.get("streaming", False),
                description=config_params.get("description", ""),
                use_case=config_params.get("use_case", ""),
                specialization=config_params.get("specialization", ""),
                reasoning_depth=config_params.get("reasoning_depth", "standard"),
                creativity_level=config_params.get("creativity_level", "balanced"),
                precision_level=config_params.get("precision_level", "high")
            )
            
            return config
            
        except Exception as e:
            logger.error(f"LLM 설정 생성 실패: {e}")
            # 폴백 설정 - Gemini로 변경
            return OntologyLLMConfig(
                provider=LLMProvider.GOOGLE,
                model="gemini-2.0-flash-lite",
                description=f"폴백 설정 for {llm_type_str}"
            )
    
    def load_profile_configs(self, profile_name: str) -> Dict[OntologyLLMType, OntologyLLMConfig]:
        """프로파일의 모든 LLM 설정 로드"""
        configs = {}
        
        # 프로파일 설정 조회
        profile_settings = self.get_profile_settings(profile_name)
        
        if not profile_settings:
            logger.warning(f"프로파일이 비어있습니다: {profile_name}")
            return configs
        
        # 각 LLM 타입별 설정 생성
        for llm_type in OntologyLLMType:
            llm_type_str = llm_type.value
            
            if llm_type_str not in profile_settings:
                logger.warning(f"프로파일에 {llm_type_str} 설정이 없습니다")
                continue
            
            setting = profile_settings[llm_type_str]
            provider = setting.get("provider", "openai")
            model_tier = setting.get("model_tier", "standard")
            
            config = self.create_llm_config(llm_type_str, provider, model_tier)
            configs[llm_type] = config
            
            logger.debug(f"✅ {llm_type_str}: {provider}/{config.model}")
        
        return configs
    
    def get_default_profile(self) -> str:
        """기본 프로파일 이름"""
        return self.config_data.get("default_profile", "all_gemini")
    
    def get_profile_description(self, profile_name: str) -> str:
        """프로파일 설명"""
        profiles = self.config_data.get("profiles", {})
        if profile_name in profiles:
            return profiles[profile_name].get("description", "")
        return ""
    
    def list_profiles_info(self) -> Dict[str, str]:
        """모든 프로파일 정보"""
        profiles = self.config_data.get("profiles", {})
        return {
            name: profile.get("description", "")
            for name, profile in profiles.items()
        }
    
    def validate_config(self) -> List[str]:
        """설정 유효성 검사"""
        errors = []
        
        # 필수 섹션 확인
        required_sections = ["providers", "profiles"]
        for section in required_sections:
            if section not in self.config_data:
                errors.append(f"필수 섹션이 없습니다: {section}")
        
        # 제공업체 설정 확인
        providers = self.config_data.get("providers", {})
        for provider_name, provider_config in providers.items():
            if "models" not in provider_config:
                errors.append(f"제공업체 {provider_name}에 모델 설정이 없습니다")
        
        # 프로파일 설정 확인
        profiles = self.config_data.get("profiles", {})
        for profile_name, profile_config in profiles.items():
            if "settings" not in profile_config:
                errors.append(f"프로파일 {profile_name}에 settings가 없습니다")
        
        return errors
    
    def save_config(self, new_config: Dict[str, Any]):
        """설정 파일 저장"""
        try:
            # 백업 생성
            if self.config_path.exists():
                backup_path = self.config_path.with_suffix('.yaml.backup')
                self.config_path.rename(backup_path)
                logger.info(f"기존 설정 백업: {backup_path}")
            
            # 새 설정 저장
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(new_config, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            self.config_data = new_config
            logger.info(f"✅ 설정 파일 저장 완료: {self.config_path}")
            
        except Exception as e:
            logger.error(f"❌ 설정 파일 저장 실패: {e}")
            raise e
    
    def add_custom_profile(self, profile_name: str, description: str, settings: Dict[str, Dict[str, str]]):
        """사용자 정의 프로파일 추가"""
        try:
            if "profiles" not in self.config_data:
                self.config_data["profiles"] = {}
            
            self.config_data["profiles"][profile_name] = {
                "description": description,
                "settings": settings
            }
            
            self.save_config(self.config_data)
            logger.info(f"✅ 사용자 정의 프로파일 추가: {profile_name}")
            
        except Exception as e:
            logger.error(f"❌ 프로파일 추가 실패: {e}")
            raise e
    
    def get_config_summary(self) -> Dict[str, Any]:
        """설정 요약 정보"""
        return {
            "config_version": self.config_data.get("config_version", "unknown"),
            "last_updated": self.config_data.get("last_updated", "unknown"),
            "default_profile": self.get_default_profile(),
            "available_providers": self.get_available_providers(),
            "available_profiles": self.get_available_profiles(),
            "total_profiles": len(self.get_available_profiles()),
            "config_path": str(self.config_path),
            "config_exists": self.config_path.exists()
        }


# 전역 인스턴스
_global_config_loader: Optional[LLMConfigLoader] = None


def get_llm_config_loader() -> LLMConfigLoader:
    """전역 설정 로더 인스턴스"""
    global _global_config_loader
    if _global_config_loader is None:
        _global_config_loader = LLMConfigLoader()
    return _global_config_loader


def reload_llm_config():
    """설정 다시 로드"""
    global _global_config_loader
    _global_config_loader = None
    return get_llm_config_loader()


logger.info("🔧 LLM 설정 로더 모듈 로드 완료!") 