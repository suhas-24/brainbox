"""
Advanced Logging System for AI Forge

Provides structured logging with multiple handlers, formatters, and
environment-based configuration.
"""

import logging
import logging.config
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_entry["user_id"] = record.user_id
        if hasattr(record, 'session_id'):
            log_entry["session_id"] = record.session_id
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id
        if hasattr(record, 'execution_time'):
            log_entry["execution_time"] = record.execution_time
        if hasattr(record, 'token_count'):
            log_entry["token_count"] = record.token_count
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False)


class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for better readability."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors for console output."""
        color = self.COLORS.get(record.levelname, '')
        reset = self.RESET
        
        # Format: [TIME] LEVEL LOGGER: MESSAGE
        formatted = f"{color}[{self.formatTime(record, '%H:%M:%S')}] {record.levelname:8} {record.name}: {record.getMessage()}{reset}"
        
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        
        return formatted


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    structured: bool = False,
    console_colors: bool = True
) -> None:
    """
    Setup comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        structured: Use JSON formatting
        console_colors: Use colored console output
    """
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if console_colors and not structured:
        console_formatter = ColoredConsoleFormatter()
    elif structured:
        console_formatter = StructuredFormatter()
    else:
        console_formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s %(name)s: %(message)s'
        )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=5,
            encoding='utf-8'
        )
        
        if structured:
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s %(name)s [%(filename)s:%(lineno)d]: %(message)s'
            )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")


class PerformanceLogger:
    """Logger for performance metrics and monitoring."""
    
    def __init__(self, name: str = "performance"):
        self.logger = get_logger(name)
    
    def log_execution_time(
        self, 
        operation: str, 
        execution_time: float, 
        **kwargs
    ) -> None:
        """Log execution time for an operation."""
        self.logger.info(
            f"Operation completed: {operation}",
            extra={
                "execution_time": execution_time,
                "operation": operation,
                **kwargs
            }
        )
    
    def log_token_usage(
        self,
        operation: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        total_cost: float = None,
        **kwargs
    ) -> None:
        """Log token usage for LLM operations."""
        self.logger.info(
            f"Token usage: {operation}",
            extra={
                "operation": operation,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "total_cost": total_cost,
                **kwargs
            }
        )
    
    def log_api_request(
        self,
        provider: str,
        endpoint: str,
        status_code: int,
        response_time: float,
        **kwargs
    ) -> None:
        """Log API request metrics."""
        level = logging.INFO if status_code < 400 else logging.ERROR
        self.logger.log(
            level,
            f"API request: {provider} {endpoint}",
            extra={
                "provider": provider,
                "endpoint": endpoint,
                "status_code": status_code,
                "response_time": response_time,
                **kwargs
            }
        )


class SecurityLogger:
    """Logger for security events and audit trails."""
    
    def __init__(self, name: str = "security"):
        self.logger = get_logger(name)
    
    def log_authentication(
        self,
        user_id: str,
        success: bool,
        ip_address: str = None,
        user_agent: str = None,
        **kwargs
    ) -> None:
        """Log authentication attempts."""
        level = logging.INFO if success else logging.WARNING
        status = "success" if success else "failure"
        
        self.logger.log(
            level,
            f"Authentication {status} for user {user_id}",
            extra={
                "user_id": user_id,
                "auth_success": success,
                "ip_address": ip_address,
                "user_agent": user_agent,
                **kwargs
            }
        )
    
    def log_authorization(
        self,
        user_id: str,
        resource: str,
        action: str,
        granted: bool,
        **kwargs
    ) -> None:
        """Log authorization decisions."""
        level = logging.INFO if granted else logging.WARNING
        status = "granted" if granted else "denied"
        
        self.logger.log(
            level,
            f"Authorization {status}: {user_id} -> {action} on {resource}",
            extra={
                "user_id": user_id,
                "resource": resource,
                "action": action,
                "access_granted": granted,
                **kwargs
            }
        )
    
    def log_suspicious_activity(
        self,
        user_id: str,
        activity: str,
        risk_score: float,
        details: Dict[str, Any],
        **kwargs
    ) -> None:
        """Log suspicious activities."""
        self.logger.warning(
            f"Suspicious activity detected: {activity}",
            extra={
                "user_id": user_id,
                "activity": activity,
                "risk_score": risk_score,
                "details": details,
                **kwargs
            }
        )


# Default configuration
def configure_default_logging():
    """Configure default logging for AI Forge applications."""
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_file = os.getenv("LOG_FILE", "./logs/app.log")
    structured = os.getenv("LOG_FORMAT", "").lower() == "json"
    
    setup_logging(
        log_level=log_level,
        log_file=log_file,
        structured=structured
    )


# Auto-configure if imported
if not logging.getLogger().handlers:
    configure_default_logging()
