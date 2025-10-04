"""
Logging service implementation for YOLO Editor.
Provides structured logging with different levels and output options.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, TextIO
from pathlib import Path
import logging
import sys
from datetime import datetime
from enum import Enum

from .interfaces import ILogger


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class LoggingService(ILogger):
    """Concrete implementation of logging service."""
    
    def __init__(self, name: str = "YOLOEditor", log_file: Optional[Path] = None, 
                 console_level: LogLevel = LogLevel.INFO, file_level: LogLevel = LogLevel.DEBUG):
        self._name = name
        self._log_file = log_file
        self._console_level = console_level
        self._file_level = file_level
        
        # Set up Python logging
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        self._logger.handlers.clear()
        
        # Set up console handler
        self._setup_console_handler()
        
        # Set up file handler if specified
        if log_file:
            self._setup_file_handler()
    
    def _setup_console_handler(self) -> None:
        """Set up console logging handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self._console_level.value))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        self._logger.addHandler(console_handler)
    
    def _setup_file_handler(self) -> None:
        """Set up file logging handler."""
        if not self._log_file:
            return
        
        try:
            # Ensure log directory exists
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(self._log_file, encoding='utf-8')
            file_handler.setLevel(getattr(logging, self._file_level.value))
            
            # Create detailed formatter for file
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            
            self._logger.addHandler(file_handler)
            
        except Exception as e:
            print(f"Failed to set up file logging: {e}", file=sys.stderr)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message."""
        extra_info = self._format_extra_info(kwargs)
        self._logger.debug(f"{message}{extra_info}")
    
    def info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        extra_info = self._format_extra_info(kwargs)
        self._logger.info(f"{message}{extra_info}")
    
    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        extra_info = self._format_extra_info(kwargs)
        self._logger.warning(f"{message}{extra_info}")
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log an error message."""
        extra_info = self._format_extra_info(kwargs)
        full_message = f"{message}{extra_info}"
        
        if exception:
            self._logger.error(full_message, exc_info=exception)
        else:
            self._logger.error(full_message)
    
    def _format_extra_info(self, kwargs: Dict[str, Any]) -> str:
        """Format additional keyword arguments into a string."""
        if not kwargs:
            return ""
        
        parts = []
        for key, value in kwargs.items():
            parts.append(f"{key}={value}")
        
        return f" [{', '.join(parts)}]"
    
    def set_console_level(self, level: LogLevel) -> None:
        """Change the console logging level."""
        self._console_level = level
        for handler in self._logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                handler.setLevel(getattr(logging, level.value))
                break
    
    def set_file_level(self, level: LogLevel) -> None:
        """Change the file logging level."""
        self._file_level = level
        for handler in self._logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.setLevel(getattr(logging, level.value))
                break
    
    def add_file_handler(self, log_file: Path, level: LogLevel = LogLevel.DEBUG) -> bool:
        """Add an additional file handler."""
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(getattr(logging, level.value))
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            
            self._logger.addHandler(file_handler)
            return True
            
        except Exception as e:
            self.error(f"Failed to add file handler for {log_file}", exception=e)
            return False
    
    def log_performance(self, operation: str, duration_ms: float, **kwargs) -> None:
        """Log performance information."""
        self.info(f"Performance: {operation} took {duration_ms:.2f}ms", **kwargs)
    
    def log_user_action(self, action: str, **kwargs) -> None:
        """Log user actions for analytics/debugging."""
        self.info(f"User action: {action}", **kwargs)
    
    def log_system_info(self, info: Dict[str, Any]) -> None:
        """Log system information."""
        self.info("System info", **info)
    
    def create_child_logger(self, name: str) -> 'LoggingService':
        """Create a child logger with the same configuration."""
        child_name = f"{self._name}.{name}"
        return LoggingService(
            name=child_name,
            log_file=self._log_file,
            console_level=self._console_level,
            file_level=self._file_level
        )


class NullLogger(ILogger):
    """Null logger implementation for testing or when logging is disabled."""
    
    def debug(self, message: str, **kwargs) -> None:
        pass
    
    def info(self, message: str, **kwargs) -> None:
        pass
    
    def warning(self, message: str, **kwargs) -> None:
        pass
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        pass


class MemoryLogger(ILogger):
    """In-memory logger for testing purposes."""
    
    def __init__(self, max_entries: int = 1000):
        self._max_entries = max_entries
        self._entries: list = []
    
    def debug(self, message: str, **kwargs) -> None:
        self._add_entry("DEBUG", message, kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        self._add_entry("INFO", message, kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        self._add_entry("WARNING", message, kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        entry_kwargs = dict(kwargs)
        if exception:
            entry_kwargs['exception'] = str(exception)
        self._add_entry("ERROR", message, entry_kwargs)
    
    def _add_entry(self, level: str, message: str, kwargs: Dict[str, Any]) -> None:
        """Add an entry to the memory log."""
        entry = {
            'timestamp': datetime.now(),
            'level': level,
            'message': message,
            'kwargs': kwargs
        }
        
        self._entries.append(entry)
        
        # Trim if necessary
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]
    
    def get_entries(self, level: Optional[str] = None) -> list:
        """Get log entries, optionally filtered by level."""
        if level:
            return [e for e in self._entries if e['level'] == level]
        return self._entries.copy()
    
    def clear(self) -> None:
        """Clear all log entries."""
        self._entries.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about log entries."""
        stats = {'DEBUG': 0, 'INFO': 0, 'WARNING': 0, 'ERROR': 0}
        for entry in self._entries:
            level = entry['level']
            if level in stats:
                stats[level] += 1
        return stats