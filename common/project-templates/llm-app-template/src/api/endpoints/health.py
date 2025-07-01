"""
Health Check API Endpoints

Provides health monitoring endpoints for the application,
including system status, provider health, and readiness checks.
"""

from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...utils import get_logger
from ..app import get_app_state


router = APIRouter()
logger = get_logger(__name__)


class HealthStatus(BaseModel):
    """Health status model."""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Application version")
    uptime: float = Field(..., description="Application uptime in seconds")
    components: Dict[str, Dict[str, Any]] = Field(..., description="Component health details")


class ReadinessStatus(BaseModel):
    """Readiness status model."""
    ready: bool = Field(..., description="Application readiness status")
    timestamp: datetime = Field(..., description="Readiness check timestamp")
    checks: Dict[str, bool] = Field(..., description="Individual readiness checks")


@router.get("/", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """
    Get application health status.
    
    Returns:
        Health status information
    """
    try:
        app_state = get_app_state()
        
        # Get component health
        components = {}
        
        # Check provider manager
        if "provider_manager" in app_state:
            provider_manager = app_state["provider_manager"]
            provider_health = await provider_manager.health_check()
            components["providers"] = {
                "status": "healthy" if provider_health["healthy"] else "unhealthy",
                "details": provider_health
            }
        else:
            components["providers"] = {
                "status": "unavailable",
                "details": {"error": "Provider manager not initialized"}
            }
        
        # Check memory manager
        if "memory_manager" in app_state:
            try:
                memory_stats = await app_state["memory_manager"].get_memory_stats()
                components["memory"] = {
                    "status": "healthy",
                    "details": {
                        "sessions": memory_stats.short_term_sessions,
                        "messages": memory_stats.short_term_messages
                    }
                }
            except Exception as e:
                components["memory"] = {
                    "status": "unhealthy",
                    "details": {"error": str(e)}
                }
        else:
            components["memory"] = {
                "status": "unavailable",
                "details": {"error": "Memory manager not initialized"}
            }
        
        # Determine overall status
        overall_status = "healthy"
        for component in components.values():
            if component["status"] != "healthy":
                overall_status = "degraded" if overall_status == "healthy" else "unhealthy"
                break
        
        return HealthStatus(
            status=overall_status,
            timestamp=datetime.now(),
            version="1.0.0",
            uptime=0.0,  # Would calculate actual uptime
            components=components
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/ready", response_model=ReadinessStatus)
async def readiness_check() -> ReadinessStatus:
    """
    Get application readiness status.
    
    Returns:
        Readiness status information
    """
    try:
        app_state = get_app_state()
        checks = {}
        
        # Check if core components are initialized
        checks["app_state"] = len(app_state) > 0
        checks["provider_manager"] = "provider_manager" in app_state
        checks["memory_manager"] = "memory_manager" in app_state
        checks["config"] = "config" in app_state
        
        # Check provider initialization
        if checks["provider_manager"]:
            provider_manager = app_state["provider_manager"]
            provider_health = await provider_manager.health_check()
            checks["providers_ready"] = provider_health.get("healthy", False)
        else:
            checks["providers_ready"] = False
        
        # Overall readiness
        ready = all(checks.values())
        
        return ReadinessStatus(
            ready=ready,
            timestamp=datetime.now(),
            checks=checks
        )
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Readiness check failed")


@router.get("/live")
async def liveness_check() -> Dict[str, str]:
    """
    Simple liveness check.
    
    Returns:
        Basic liveness confirmation
    """
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat()
    }
