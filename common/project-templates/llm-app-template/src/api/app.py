"""
FastAPI Application Factory for AI Forge

Creates and configures the main FastAPI application with proper
lifecycle management, middleware setup, and route registration.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ..config import AppConfig
from ..providers import ProviderManager
from ..memory import MemoryManager
from ..utils import get_logger
from .middleware import setup_middleware
from .routes import setup_routes


# Global application state
app_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger = get_logger(__name__)
    
    try:
        # Startup
        logger.info("Starting AI Forge application...")
        
        # Initialize core components
        config = AppConfig()
        provider_manager = ProviderManager(config)
        memory_manager = MemoryManager()
        
        # Store in application state
        app_state.update({
            "config": config,
            "provider_manager": provider_manager,
            "memory_manager": memory_manager,
            "logger": logger
        })
        
        # Initialize providers
        await provider_manager.initialize()
        
        logger.info("AI Forge application started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down AI Forge application...")
        
        # Cleanup providers
        if "provider_manager" in app_state:
            await app_state["provider_manager"].cleanup()
        
        # Clear state
        app_state.clear()
        
        logger.info("AI Forge application shutdown complete")


def create_app(config: AppConfig = None) -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Args:
        config: Application configuration (optional)
        
    Returns:
        Configured FastAPI application
    """
    if config is None:
        config = AppConfig()
    
    # Create FastAPI app
    app = FastAPI(
        title="BrainBox",
        description="Complete AI Intelligence Framework - Your AI brain in a box",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if config.environment != "production" else None,
        redoc_url="/redoc" if config.environment != "production" else None
    )
    
    # Setup middleware
    setup_middleware(app, config)
    
    # Setup routes
    setup_routes(app)
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger = get_logger(__name__)
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": "An unexpected error occurred"
            }
        )
    
    return app


def get_app_state() -> Dict[str, Any]:
    """Get application state."""
    return app_state
