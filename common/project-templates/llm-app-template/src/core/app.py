"""
FastAPI Application Factory for AI Forge

Creates and configures the main FastAPI application with all middleware,
routes, and dependencies properly initialized.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .config import get_settings
from .llm_manager import LLMManager
from ..utils import get_logger, setup_logging
from ..memory import MemoryManager
from ..agents import AgentCoordinator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger = get_logger(__name__)
    settings = get_settings()
    
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    # Initialize core components
    app.state.llm_manager = LLMManager()
    app.state.memory_manager = MemoryManager()
    app.state.agent_coordinator = AgentCoordinator()
    
    # Health check for LLM connections
    try:
        providers = app.state.llm_manager.get_available_providers()
        logger.info(f"Available LLM providers: {providers}")
    except Exception as e:
        logger.warning(f"LLM provider initialization issue: {e}")
    
    logger.info("Application startup complete")
    
    yield
    
    # Cleanup
    logger.info("Application shutdown initiated")


async def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    # Setup logging first
    setup_logging(
        log_level=settings.log_level,
        log_file="./logs/app.log" if not settings.debug else None,
        structured=False
    )
    
    logger = get_logger(__name__)
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Advanced LLM Application built with AI Forge",
        debug=settings.debug,
        lifespan=lifespan
    )
    
    # Add middleware
    await _add_middleware(app, settings)
    
    # Add routes
    await _add_routes(app)
    
    # Add exception handlers
    await _add_exception_handlers(app)
    
    logger.info("FastAPI application created successfully")
    return app


async def _add_middleware(app: FastAPI, settings):
    """Add middleware to the application."""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        logger = get_logger("api.requests")
        start_time = asyncio.get_event_loop().time()
        
        # Log request
        logger.info(f"{request.method} {request.url.path}", extra={
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else "unknown"
        })
        
        try:
            response = await call_next(request)
            
            # Log response
            process_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"Response {response.status_code}", extra={
                "status_code": response.status_code,
                "process_time": process_time,
                "path": request.url.path
            })
            
            response.headers["X-Process-Time"] = str(process_time)
            return response
        
        except Exception as e:
            process_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Request failed: {e}", extra={
                "error": str(e),
                "process_time": process_time,
                "path": request.url.path
            })
            raise


async def _add_routes(app: FastAPI):
    """Add API routes to the application."""
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "AI Forge Application",
            "status": "running",
            "version": get_settings().app_version
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        settings = get_settings()
        
        health_status = {
            "status": "healthy",
            "timestamp": asyncio.get_event_loop().time(),
            "version": settings.app_version,
            "components": {}
        }
        
        # Check LLM manager
        try:
            if hasattr(app.state, 'llm_manager'):
                llm_status = app.state.llm_manager.get_health_status()
                health_status["components"]["llm"] = llm_status
            else:
                health_status["components"]["llm"] = {"status": "not_initialized"}
        except Exception as e:
            health_status["components"]["llm"] = {"status": "error", "error": str(e)}
        
        # Check memory manager
        try:
            if hasattr(app.state, 'memory_manager'):
                health_status["components"]["memory"] = {"status": "healthy"}
            else:
                health_status["components"]["memory"] = {"status": "not_initialized"}
        except Exception as e:
            health_status["components"]["memory"] = {"status": "error", "error": str(e)}
        
        # Determine overall status
        component_statuses = [comp.get("status", "unknown") for comp in health_status["components"].values()]
        if any(status == "error" for status in component_statuses):
            health_status["status"] = "unhealthy"
        elif any(status in ["degraded", "not_initialized"] for status in component_statuses):
            health_status["status"] = "degraded"
        
        return health_status
    
    @app.post("/api/chat")
    async def chat_completion(request: Dict[str, Any]):
        """Chat completion endpoint."""
        try:
            message = request.get("message", "")
            model = request.get("model")
            provider = request.get("provider")
            session_id = request.get("session_id", "default")
            
            if not message:
                raise HTTPException(status_code=400, detail="Message is required")
            
            # Use LLM manager to generate response
            llm_manager = app.state.llm_manager
            
            if provider and model:
                response = await llm_manager.generate(
                    message, provider=provider, model=model
                )
            else:
                response = await llm_manager.generate_with_fallback(message)
            
            return {
                "response": response,
                "session_id": session_id,
                "model_used": model or "default",
                "provider_used": provider or "default"
            }
        
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Chat completion error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/providers")
    async def get_providers():
        """Get available LLM providers."""
        try:
            llm_manager = app.state.llm_manager
            providers = llm_manager.get_available_providers()
            
            return {
                "providers": providers,
                "count": len(providers)
            }
        
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Error getting providers: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/stats")
    async def get_stats():
        """Get application statistics."""
        try:
            stats = {
                "llm": {},
                "memory": {},
                "agents": {}
            }
            
            # LLM stats
            if hasattr(app.state, 'llm_manager'):
                stats["llm"] = app.state.llm_manager.get_usage_stats()
            
            # Memory stats (placeholder)
            if hasattr(app.state, 'memory_manager'):
                stats["memory"] = {"status": "active"}
            
            # Agent stats (placeholder)
            if hasattr(app.state, 'agent_coordinator'):
                stats["agents"] = {"status": "active"}
            
            return stats
        
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Error getting stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))


async def _add_exception_handlers(app: FastAPI):
    """Add global exception handlers."""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        logger = get_logger("api.errors")
        logger.warning(f"HTTP {exc.status_code}: {exc.detail}", extra={
            "status_code": exc.status_code,
            "path": request.url.path,
            "method": request.method
        })
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "path": request.url.path
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger = get_logger("api.errors")
        logger.error(f"Unhandled exception: {exc}", extra={
            "path": request.url.path,
            "method": request.method,
            "error_type": type(exc).__name__
        })
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "status_code": 500,
                "path": request.url.path
            }
        )


# For direct uvicorn usage
app = None

async def get_app():
    """Get or create application instance."""
    global app
    if app is None:
        app = await create_app()
    return app
