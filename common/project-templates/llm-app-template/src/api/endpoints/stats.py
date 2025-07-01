"""
Statistics API Endpoints

Provides endpoints for collecting and retrieving statistics on
LLM usage, performance, and memory utilization.
"""

from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...utils import get_logger
from ..app import get_app_state


router = APIRouter()
logger = get_logger(__name__)


class UsageStats(BaseModel):
    """LLM usage statistics model."""
    total_requests: int = Field(..., description="Total number of requests")
    total_tokens: int = Field(..., description="Total number of tokens used")
    average_latency: float = Field(..., description="Average latency in ms")
    provider_breakdown: Dict[str, Dict[str, Any]] = Field(..., description="Usage breakdown by provider")


@router.get("/usage", response_model=UsageStats)
async def get_usage_stats() -> UsageStats:
    """
    Get LLM usage statistics.

    Returns:
        Usage statistics
    """
    try:
        # Placeholder example - Actual implementation would gather real statistics
        usage_stats = UsageStats(
            total_requests=1000,
            total_tokens=250000,
            average_latency=120.5,
            provider_breakdown={
                "provider1": {
                    "requests": 600,
                    "tokens": 150000
                },
                "provider2": {
                    "requests": 400,
                    "tokens": 100000
                }
            }
        )

        return usage_stats

    except Exception as e:
        logger.error(f"Failed to get usage stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance")
async def get_performance_stats() -> Dict[str, Any]:
    """
    Get performance statistics.

    Returns:
        Performance metrics
    """
    try:
        # Placeholder example - Real implementation would gather performance metrics
        perf_stats = {
            "cpu_usage": "23%",
            "memory_usage": "1.5 GB",
            "active_sessions": 42
        }

        return perf_stats

    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory")
async def get_memory_stats() -> Dict[str, Any]:
    """
    Get system memory statistics.

    Returns:
        Memory statistics
    """
    try:
        app_state = get_app_state()
        memory_manager = app_state["memory_manager"]

        stats = await memory_manager.get_memory_stats()

        memory_stats = {
            "short_term_sessions": stats.short_term_sessions,
            "short_term_messages": stats.short_term_messages,
            "long_term_entries": stats.long_term_entries,
            "vector_embeddings": stats.vector_embeddings,
            "total_memory_mb": stats.total_memory_mb
        }

        return memory_stats

    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
