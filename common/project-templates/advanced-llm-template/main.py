#!/usr/bin/env python3
"""
AI Forge Advanced LLM Template - Main Application Entry Point

This is the main entry point for the advanced LLM application built with the AI Forge framework.
It provides a production-ready, scalable foundation for building sophisticated AI applications.
"""

import asyncio
import logging
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.app import create_app
from src.core.config import get_settings
from src.utils.logger import setup_logging


async def main():
    """Main application entry point."""
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        settings = get_settings()
        logger.info(f"Starting {settings.app_name} v{settings.app_version}")
        
        # Create FastAPI application
        app = await create_app()
        
        # Configure uvicorn
        config = uvicorn.Config(
            app=app,
            host=settings.api_host,
            port=settings.api_port,
            log_level=settings.log_level.lower(),
            reload=settings.debug,
            workers=1 if settings.debug else settings.api_workers,
            access_log=True,
            use_colors=True,
        )
        
        # Start the server
        server = uvicorn.Server(config)
        logger.info(f"Server starting on {settings.api_host}:{settings.api_port}")
        await server.serve()
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)


def run_server():
    """Run the server using uvicorn directly (for development)."""
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        workers=1 if settings.debug else settings.api_workers,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Forge Advanced LLM Application")
    parser.add_argument(
        "--mode",
        choices=["server", "cli", "worker"],
        default="server",
        help="Application mode (default: server)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    if args.mode == "server":
        if args.debug or len(sys.argv) == 1:  # Default behavior or explicit debug
            run_server()
        else:
            asyncio.run(main())
    elif args.mode == "cli":
        # CLI mode for command-line interactions
        from src.cli.main import run_cli
        asyncio.run(run_cli())
    elif args.mode == "worker":
        # Background worker mode for async tasks
        from src.workers.main import run_worker
        asyncio.run(run_worker())
    else:
        parser.print_help()
        sys.exit(1)
