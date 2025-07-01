"""
Application configuration management.
"""

import os
from functools import lru_cache
from typing import Optional, List

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    app_name: str = Field(default="BrainBox", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="BRAINBOX_LOG_LEVEL")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # LLM API Keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    
    # Database
    database_url: str = Field(default="sqlite:///./data/app.db", env="DATABASE_URL")
    
    # Security
    secret_key: str = Field(default="your-secret-key", env="SECRET_KEY")
    jwt_secret: str = Field(default="your-jwt-secret", env="JWT_SECRET")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS"
    )
    
    # Model Configuration
    default_model_provider: str = Field(default="openai", env="DEFAULT_MODEL_PROVIDER")
    default_model: str = Field(default="gpt-4", env="DEFAULT_MODEL")
    default_temperature: float = Field(default=0.7, env="DEFAULT_TEMPERATURE")
    default_max_tokens: int = Field(default=1000, env="DEFAULT_MAX_TOKENS")
    
    # Performance
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    
    # Feature Flags
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    enable_rate_limiting: bool = Field(default=True, env="ENABLE_RATE_LIMITING")
    enable_analytics: bool = Field(default=False, env="ENABLE_ANALYTICS")
    enable_webhooks: bool = Field(default=False, env="ENABLE_WEBHOOKS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
