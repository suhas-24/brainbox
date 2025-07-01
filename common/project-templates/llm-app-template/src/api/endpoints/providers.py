"""
Provider Management API Endpoints

Provides endpoints for managing LLM providers including listing available
providers, getting provider status, and configuring provider settings.
"""

from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...utils import get_logger
from ..app import get_app_state


router = APIRouter()
logger = get_logger(__name__)


class ProviderInfo(BaseModel):
    """Provider information model."""
    name: str = Field(..., description="Provider name")
    enabled: bool = Field(..., description="Whether provider is enabled")
    models: List[str] = Field(..., description="Available models")
    health: str = Field(..., description="Provider health status")
    config: Dict[str, Any] = Field(..., description="Provider configuration")


class ProviderHealth(BaseModel):
    """Provider health model."""
    name: str = Field(..., description="Provider name")
    healthy: bool = Field(..., description="Health status")
    latency: Optional[float] = Field(None, description="Response latency in ms")
    error: Optional[str] = Field(None, description="Error message if unhealthy")


@router.get("/", response_model=List[ProviderInfo])
async def list_providers() -> List[ProviderInfo]:
    """
    List all available providers.
    
    Returns:
        List of provider information
    """
    try:
        app_state = get_app_state()
        provider_manager = app_state["provider_manager"]
        
        providers_info = []
        provider_health = await provider_manager.health_check()
        
        for provider_name, provider in provider_manager.providers.items():
            # Get provider models
            models = []
            try:
                models = await provider.list_models()
            except Exception as e:
                logger.warning(f"Failed to get models for {provider_name}: {e}")
            
            # Get health status
            health_status = "unknown"
            if provider_name in provider_health.get("providers", {}):
                provider_health_info = provider_health["providers"][provider_name]
                health_status = "healthy" if provider_health_info.get("healthy", False) else "unhealthy"
            
            providers_info.append(ProviderInfo(
                name=provider_name,
                enabled=provider.enabled,
                models=models,
                health=health_status,
                config={
                    "max_tokens": getattr(provider, "max_tokens", None),
                    "timeout": getattr(provider, "timeout", None),
                    "rate_limit": getattr(provider, "rate_limit", None)
                }
            ))
        
        return providers_info
        
    except Exception as e:
        logger.error(f"Failed to list providers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{provider_name}", response_model=ProviderInfo)
async def get_provider(provider_name: str) -> ProviderInfo:
    """
    Get information about a specific provider.
    
    Args:
        provider_name: Name of the provider
        
    Returns:
        Provider information
    """
    try:
        app_state = get_app_state()
        provider_manager = app_state["provider_manager"]
        
        if provider_name not in provider_manager.providers:
            raise HTTPException(status_code=404, detail=f"Provider {provider_name} not found")
        
        provider = provider_manager.providers[provider_name]
        
        # Get provider models
        models = []
        try:
            models = await provider.list_models()
        except Exception as e:
            logger.warning(f"Failed to get models for {provider_name}: {e}")
        
        # Get health status
        provider_health = await provider_manager.health_check()
        health_status = "unknown"
        if provider_name in provider_health.get("providers", {}):
            provider_health_info = provider_health["providers"][provider_name]
            health_status = "healthy" if provider_health_info.get("healthy", False) else "unhealthy"
        
        return ProviderInfo(
            name=provider_name,
            enabled=provider.enabled,
            models=models,
            health=health_status,
            config={
                "max_tokens": getattr(provider, "max_tokens", None),
                "timeout": getattr(provider, "timeout", None),
                "rate_limit": getattr(provider, "rate_limit", None)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get provider {provider_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{provider_name}/health", response_model=ProviderHealth)
async def get_provider_health(provider_name: str) -> ProviderHealth:
    """
    Get health status for a specific provider.
    
    Args:
        provider_name: Name of the provider
        
    Returns:
        Provider health information
    """
    try:
        app_state = get_app_state()
        provider_manager = app_state["provider_manager"]
        
        if provider_name not in provider_manager.providers:
            raise HTTPException(status_code=404, detail=f"Provider {provider_name} not found")
        
        # Get health status
        provider_health = await provider_manager.health_check()
        
        if provider_name in provider_health.get("providers", {}):
            health_info = provider_health["providers"][provider_name]
            return ProviderHealth(
                name=provider_name,
                healthy=health_info.get("healthy", False),
                latency=health_info.get("latency"),
                error=health_info.get("error")
            )
        else:
            return ProviderHealth(
                name=provider_name,
                healthy=False,
                error="Health information not available"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get provider health for {provider_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{provider_name}/models")
async def get_provider_models(provider_name: str) -> List[str]:
    """
    Get available models for a specific provider.
    
    Args:
        provider_name: Name of the provider
        
    Returns:
        List of available models
    """
    try:
        app_state = get_app_state()
        provider_manager = app_state["provider_manager"]
        
        if provider_name not in provider_manager.providers:
            raise HTTPException(status_code=404, detail=f"Provider {provider_name} not found")
        
        provider = provider_manager.providers[provider_name]
        models = await provider.list_models()
        
        return models
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get models for provider {provider_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{provider_name}/enable")
async def enable_provider(provider_name: str) -> Dict[str, str]:
    """
    Enable a provider.
    
    Args:
        provider_name: Name of the provider
        
    Returns:
        Success confirmation
    """
    try:
        app_state = get_app_state()
        provider_manager = app_state["provider_manager"]
        
        if provider_name not in provider_manager.providers:
            raise HTTPException(status_code=404, detail=f"Provider {provider_name} not found")
        
        provider = provider_manager.providers[provider_name]
        provider.enabled = True
        
        logger.info(f"Provider {provider_name} enabled")
        
        return {"status": "success", "message": f"Provider {provider_name} enabled"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to enable provider {provider_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{provider_name}/disable")
async def disable_provider(provider_name: str) -> Dict[str, str]:
    """
    Disable a provider.
    
    Args:
        provider_name: Name of the provider
        
    Returns:
        Success confirmation
    """
    try:
        app_state = get_app_state()
        provider_manager = app_state["provider_manager"]
        
        if provider_name not in provider_manager.providers:
            raise HTTPException(status_code=404, detail=f"Provider {provider_name} not found")
        
        provider = provider_manager.providers[provider_name]
        provider.enabled = False
        
        logger.info(f"Provider {provider_name} disabled")
        
        return {"status": "success", "message": f"Provider {provider_name} disabled"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to disable provider {provider_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
