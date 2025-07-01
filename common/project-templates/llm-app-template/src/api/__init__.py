"""
FastAPI Application Layer for AI Forge

Provides RESTful API endpoints for LLM interactions, memory management,
provider controls, health monitoring, and system statistics.
"""

from .app import create_app
from .middleware import setup_middleware
from .routes import setup_routes

__all__ = [
    "create_app",
    "setup_middleware", 
    "setup_routes"
]
