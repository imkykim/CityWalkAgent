"""
Structured logging utilities for CityWalkAgent
"""

import logging
import json
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path


class StructuredLogger:
    """
    Structured logger that outputs JSON-formatted logs

    Features:
    - JSON output for easy parsing
    - Context management
    - Level-based filtering
    - File and console output
    """

    def __init__(
        self,
        name: str,
        log_file: Optional[Path] = None,
        level: int = logging.INFO
    ):
        """
        Initialize structured logger

        Args:
            name: Logger name
            log_file: Optional log file path
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(console_handler)

        # File handler (if specified)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(StructuredFormatter())
            self.logger.addHandler(file_handler)

    def log(
        self,
        level: str,
        message: str,
        **context: Any
    ) -> None:
        """
        Log message with structured context

        Args:
            level: Log level ("debug", "info", "warning", "error", "critical")
            message: Log message
            **context: Additional context fields
        """
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": level.upper(),
            "message": message,
            **context
        }

        log_method = getattr(self.logger, level.lower())
        log_method(json.dumps(log_data, ensure_ascii=False))

    def debug(self, message: str, **context: Any) -> None:
        """Log debug message"""
        self.log("debug", message, **context)

    def info(self, message: str, **context: Any) -> None:
        """Log info message"""
        self.log("info", message, **context)

    def warning(self, message: str, **context: Any) -> None:
        """Log warning message"""
        self.log("warning", message, **context)

    def error(self, message: str, **context: Any) -> None:
        """Log error message"""
        self.log("error", message, **context)

    def critical(self, message: str, **context: Any) -> None:
        """Log critical message"""
        self.log("critical", message, **context)


class StructuredFormatter(logging.Formatter):
    """Formatter that preserves JSON structure"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record"""
        try:
            # Try to parse message as JSON
            data = json.loads(record.getMessage())
            return json.dumps(data, ensure_ascii=False)
        except json.JSONDecodeError:
            # Fallback to plain text
            return record.getMessage()


# Global logger instance
_logger: Optional[StructuredLogger] = None


def get_logger(name: str = "city_walk_agent") -> StructuredLogger:
    """
    Get global logger instance

    Args:
        name: Logger name

    Returns:
        StructuredLogger instance
    """
    global _logger

    if _logger is None:
        _logger = StructuredLogger(name)

    return _logger
