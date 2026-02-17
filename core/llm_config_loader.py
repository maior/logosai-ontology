"""
🔧 LLM Configuration Loader

Module that reads YAML configuration files and manages LLM settings
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
    """Model mapping information"""
    provider: str
    model_tier: str
    model_name: str
    reason: Optional[str] = None


class LLMConfigLoader:
    """LLM configuration loader"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize"""
        if config_path is None:
            # Default configuration file path
            config_path = Path(__file__).parent.parent / "config" / "llm_config.yaml"
        
        self.config_path = Path(config_path)
        self.config_data: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration file"""
        try:
            if not self.config_path.exists():
                logger.warning(f"Configuration file not found: {self.config_path}")
                self._create_default_config()
                return
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_data = yaml.safe_load(f)
            
            logger.info(f"✅ LLM configuration file loaded: {self.config_path}")
            
        except Exception as e:
            logger.error(f"❌ Configuration file load failed: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration"""
        logger.info("Creating default configuration...")
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
                    "description": "Default configuration (Gemini)",
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
        """List of available providers"""
        return list(self.config_data.get("providers", {}).keys())
    
    def get_available_profiles(self) -> List[str]:
        """List of available profiles"""
        return list(self.config_data.get("profiles", {}).keys())
    
    def get_provider_models(self, provider: str) -> Dict[str, str]:
        """Model list for provider"""
        providers = self.config_data.get("providers", {})
        if provider not in providers:
            return {}
        
        return providers[provider].get("models", {})
    
    def get_model_name(self, provider: str, model_tier: str) -> str:
        """Get model name"""
        models = self.get_provider_models(provider)
        return models.get(model_tier, models.get("standard", "gpt-4o-mini"))
    
    def get_profile_settings(self, profile_name: str) -> Dict[str, Any]:
        """Get profile settings"""
        profiles = self.config_data.get("profiles", {})
        if profile_name not in profiles:
            logger.warning(f"Profile not found: {profile_name}")
            return {}
        
        return profiles[profile_name].get("settings", {})
    
    def get_ontology_llm_defaults(self, llm_type: str) -> Dict[str, Any]:
        """Default settings by ontology LLM type"""
        defaults = self.config_data.get("ontology_llm_defaults", {})
        return defaults.get(llm_type, {})
    
    def create_llm_config(self, llm_type_str: str, provider: str, model_tier: str) -> OntologyLLMConfig:
        """Create LLM configuration object"""
        try:
            # Convert provider to enum
            provider_enum = LLMProvider(provider.lower())
            
            # Get model name
            model_name = self.get_model_name(provider, model_tier)
            
            # Get default settings
            defaults = self.get_ontology_llm_defaults(llm_type_str)
            provider_defaults = self.config_data.get("providers", {}).get(provider, {}).get("default_params", {})
            
            # Merge settings (priority: ontology_llm_defaults > provider_defaults)
            config_params = {}
            config_params.update(provider_defaults)
            config_params.update(defaults)
            
            # Create OntologyLLMConfig
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
            logger.error(f"LLM configuration creation failed: {e}")
            # Fallback configuration - switch to Gemini
            return OntologyLLMConfig(
                provider=LLMProvider.GOOGLE,
                model="gemini-2.0-flash-lite",
                description=f"Fallback config for {llm_type_str}"
            )
    
    def load_profile_configs(self, profile_name: str) -> Dict[OntologyLLMType, OntologyLLMConfig]:
        """Load all LLM configurations for a profile"""
        configs = {}
        
        # Get profile settings
        profile_settings = self.get_profile_settings(profile_name)
        
        if not profile_settings:
            logger.warning(f"Profile is empty: {profile_name}")
            return configs
        
        # Create configuration for each LLM type
        for llm_type in OntologyLLMType:
            llm_type_str = llm_type.value
            
            if llm_type_str not in profile_settings:
                logger.warning(f"Profile has no {llm_type_str} configuration")
                continue
            
            setting = profile_settings[llm_type_str]
            provider = setting.get("provider", "openai")
            model_tier = setting.get("model_tier", "standard")
            
            config = self.create_llm_config(llm_type_str, provider, model_tier)
            configs[llm_type] = config
            
            logger.debug(f"✅ {llm_type_str}: {provider}/{config.model}")
        
        return configs
    
    def get_default_profile(self) -> str:
        """Default profile name"""
        return self.config_data.get("default_profile", "all_gemini")
    
    def get_profile_description(self, profile_name: str) -> str:
        """Profile description"""
        profiles = self.config_data.get("profiles", {})
        if profile_name in profiles:
            return profiles[profile_name].get("description", "")
        return ""
    
    def list_profiles_info(self) -> Dict[str, str]:
        """All profile information"""
        profiles = self.config_data.get("profiles", {})
        return {
            name: profile.get("description", "")
            for name, profile in profiles.items()
        }
    
    def validate_config(self) -> List[str]:
        """Validate configuration"""
        errors = []
        
        # Check required sections
        required_sections = ["providers", "profiles"]
        for section in required_sections:
            if section not in self.config_data:
                errors.append(f"Required section missing: {section}")
        
        # Check provider configuration
        providers = self.config_data.get("providers", {})
        for provider_name, provider_config in providers.items():
            if "models" not in provider_config:
                errors.append(f"Provider {provider_name} has no model configuration")
        
        # Check profile configuration
        profiles = self.config_data.get("profiles", {})
        for profile_name, profile_config in profiles.items():
            if "settings" not in profile_config:
                errors.append(f"Profile {profile_name} has no settings")
        
        return errors
    
    def save_config(self, new_config: Dict[str, Any]):
        """Save configuration file"""
        try:
            # Create backup
            if self.config_path.exists():
                backup_path = self.config_path.with_suffix('.yaml.backup')
                self.config_path.rename(backup_path)
                logger.info(f"Existing configuration backed up: {backup_path}")
            
            # Save new configuration
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(new_config, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            self.config_data = new_config
            logger.info(f"✅ Configuration file saved: {self.config_path}")
            
        except Exception as e:
            logger.error(f"❌ Configuration file save failed: {e}")
            raise e
    
    def add_custom_profile(self, profile_name: str, description: str, settings: Dict[str, Dict[str, str]]):
        """Add custom profile"""
        try:
            if "profiles" not in self.config_data:
                self.config_data["profiles"] = {}
            
            self.config_data["profiles"][profile_name] = {
                "description": description,
                "settings": settings
            }
            
            self.save_config(self.config_data)
            logger.info(f"✅ Custom profile added: {profile_name}")
            
        except Exception as e:
            logger.error(f"❌ Profile addition failed: {e}")
            raise e
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Configuration summary information"""
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


# Global instance
_global_config_loader: Optional[LLMConfigLoader] = None


def get_llm_config_loader() -> LLMConfigLoader:
    """Global configuration loader instance"""
    global _global_config_loader
    if _global_config_loader is None:
        _global_config_loader = LLMConfigLoader()
    return _global_config_loader


def reload_llm_config():
    """Reload configuration"""
    global _global_config_loader
    _global_config_loader = None
    return get_llm_config_loader()


logger.info("🔧 LLM configuration loader module loaded!") 