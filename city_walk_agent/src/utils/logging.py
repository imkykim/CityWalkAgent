"""
Structured logging utilities for CityWalkAgent
"""

import logging
import json
from typing import Any, Dict, Optional
from pathlib import Path


class StructuredLogger:
    """
    Structured logger that outputs human-readable logs while preserving context.
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
        formatted_context = self._format_context(context)
        full_message = message if not formatted_context else f"{message} | {formatted_context}"

        log_method = getattr(self.logger, level.lower())
        log_method(full_message)

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

    @staticmethod
    def _format_context_value(value: Any) -> str:
        """Return a readable string for context values."""
        if isinstance(value, str):
            return value if " " not in value else f"\"{value}\""
        if isinstance(value, (int, float, bool)) or value is None:
            return str(value)
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return repr(value)

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Return formatted key=value pairs for log context."""
        if not context:
            return ""

        pairs = [
            f"{key}={self._format_context_value(value)}"
            for key, value in sorted(context.items())
        ]
        return " ".join(pairs)


class StructuredFormatter(logging.Formatter):
    """Formatter that provides consistent human-friendly output."""

    def __init__(self) -> None:
        super().__init__(
            fmt="[%(asctime)s] %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


# Named logger cache
_loggers: Dict[str, StructuredLogger] = {}


def get_logger(name: str = "city_walk_agent") -> StructuredLogger:
    """
    Get global logger instance

    Args:
        name: Logger name

    Returns:
        StructuredLogger instance
    """
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name)

    return _loggers[name]
