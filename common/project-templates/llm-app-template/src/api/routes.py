"""
FastAPI Routes Configuration for AI Forge

Sets up all API endpoints including chat completions, health checks,
provider management, memory operations, and system statistics.
"""

from fastapi import FastAPI, APIRouter

from .endpoints import (
    chat,
    health,
    providers,
    memory,
    stats
)


def setup_routes(app: FastAPI) -> None:
    """
    Setup API routes for the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    # Create main API router
    api_router = APIRouter(prefix="/api/v1")
    
    # Include endpoint routers
    api_router.include_router(chat.router, prefix="/chat", tags=["Chat"])
    api_router.include_router(health.router, prefix="/health", tags=["Health"])
    api_router.include_router(providers.router, prefix="/providers", tags=["Providers"])
    api_router.include_router(memory.router, prefix="/memory", tags=["Memory"])
    api_router.include_router(stats.router, prefix="/stats", tags=["Statistics"])
    
    # Add API router to app
    app.include_router(api_router)
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "BrainBox",
            "description": "Complete AI Intelligence Framework - Your AI brain in a box",
            "version": "1.0.0",
            "docs": "/docs",
            "api": "/api/v1"
        }
