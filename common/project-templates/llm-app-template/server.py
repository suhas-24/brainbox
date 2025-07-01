#!/usr/bin/env python3
"""
AI Forge Server

Main server script for running the AI Forge FastAPI application.
"""

import os
import sys
import argparse
import uvicorn
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.api import create_app
from src.config import AppConfig
from src.utils import get_logger


def main():
    """Main server entry point."""
    parser = argparse.ArgumentParser(description="AI Forge Server")
    parser.add_argument(
        "--host", 
        default="127.0.0.1", 
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level", 
        default="info", 
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level (default: info)"
    )
    parser.add_argument(
        "--env", 
        default="development",
        choices=["development", "staging", "production"],
        help="Environment (default: development)"
    )
    
    args = parser.parse_args()
    
    # Set environment
    os.environ["AI_FORGE_ENVIRONMENT"] = args.env
    
    # Initialize logger
    logger = get_logger(__name__)
    logger.info(f"Starting AI Forge server in {args.env} mode")
    
    try:
        # Create FastAPI app
        config = AppConfig()
        app = create_app(config)
        
        # Run server
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
