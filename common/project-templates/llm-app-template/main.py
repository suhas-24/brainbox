#!/usr/bin/env python3
"""
Main application entry point.
"""

import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.app import create_app
from src.core.config import get_settings


def main():
    """Main application entry point."""
    settings = get_settings()
    
    # Create and configure the application
    app = create_app()
    
    # Run the application
    if settings.debug:
        import uvicorn
        uvicorn.run(
            "main:app",
            host=settings.host,
            port=settings.port,
            reload=True,
            log_level=settings.log_level.lower()
        )
    else:
        import uvicorn
        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level.lower()
        )


if __name__ == "__main__":
    main()
