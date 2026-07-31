"""Konfigurasi logging untuk aplikasi."""

import logging
import sys
import json
import re
import psutil
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from queue import Queue


class ColoredFormatter(logging.Formatter):
    """Formatter dengan warna untuk output console."""

    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'

    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname:8s}{self.RESET}"
        return super().format(record)


class JsonFormatter(logging.Formatter):
    """Formatter untuk output JSON yang mudah di-parse."""

    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if any
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        
        return json.dumps(log_data, ensure_ascii=False)


class SensitiveDataFilter(logging.Filter):
    """Filter untuk redact data sensitif dari log messages."""
    
    # Patterns untuk sensitive data
    SENSITIVE_PATTERNS = [
        # Credentials
        (r'(password|passwd|pwd)=[^\s&]+', r'\1=***REDACTED***'),
        (r'(token|api_key|apikey)=[^\s&]+', r'\1=***REDACTED***'),
        (r'(secret|auth|authorization)=[^\s&]+', r'\1=***REDACTED***'),
        (r'Bearer\s+[A-Za-z0-9\-._~+/]+=*', r'Bearer ***REDACTED***'),
        
        # Email partial redaction: user@example.com -> u***@example.com
        (r'\b([a-zA-Z])[a-zA-Z0-9._%+-]*@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b', r'\1***@\2'),
        
        # Path redaction - Windows user paths: C:\Users\username\ -> C:\Users\***\
        (r'(?i)([C-Z]:\\Users\\)([^\\]+)(\\)', r'\1***\3'),
        
        # Path redaction - Linux home paths: /home/username/ -> /home/***/
        (r'(/home/)([^/]+)', r'\1***'),
        
        # Session/temp IDs in paths: session_abc123, tmp_xyz789 -> session_***, tmp_***
        (r'(session|temp|tmp)_[a-zA-Z0-9]+', r'\1_***'),
    ]
    
    def filter(self, record):
        """Redact sensitive data from log message."""
        if isinstance(record.msg, str):
            for pattern, replacement in self.SENSITIVE_PATTERNS:
                record.msg = re.sub(pattern, replacement, record.msg, flags=re.IGNORECASE)
        
        # Also redact from args if present
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: self._redact_value(v) for k, v in record.args.items()}
            elif isinstance(record.args, (tuple, list)):
                record.args = tuple(self._redact_value(arg) for arg in record.args)
        
        return True
    
    def _redact_value(self, value):
        """Redact sensitive value if it's a string."""
        if isinstance(value, str):
            for pattern, replacement in self.SENSITIVE_PATTERNS:
                value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
        return value


def setup_logging(
    log_level=logging.INFO,
    log_to_file=True,
    log_dir="logs",
    module_levels=None,
    enable_async=True
):
    """
    Setup konfigurasi logging aplikasi.
    
    Args:
        log_level: Default log level untuk root logger
        log_to_file: Apakah logging ke file diaktifkan
        log_dir: Direktori untuk menyimpan file log
        module_levels: Dict untuk mengatur log level per module
                      Contoh: {'core.detection': logging.DEBUG, 'ui': logging.WARNING}
        enable_async: Gunakan async logging untuk performance (default: True)
    """
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    # Add sensitive data filter to root logger
    sensitive_filter = SensitiveDataFilter()
    root_logger.addFilter(sensitive_filter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    console_format = ColoredFormatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    if log_to_file:
        # Use static filename - rotation will create .1, .2, etc. automatically
        log_file = log_path / "canopy_app.log"

        # RotatingFileHandler: rotates when file reaches maxBytes
        # Creates backup files: canopy_app.log.1, canopy_app.log.2, ..., canopy_app.log.100
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=1 * 1024 * 1024,  # 1MB per file
            backupCount=100,            # Keep 100 most recent log files
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)

        # Use JSON formatter for file output (easier parsing)
        file_format = JsonFormatter(datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_format)
        
        # Setup async logging if enabled
        if enable_async:
            # Create queue for async logging
            log_queue = Queue(-1)  # Unlimited size queue
            queue_handler = QueueHandler(log_queue)
            queue_handler.setLevel(logging.DEBUG)
            
            # QueueListener processes logs in background thread
            queue_listener = QueueListener(log_queue, file_handler, respect_handler_level=True)
            queue_listener.start()
            
            root_logger.addHandler(queue_handler)
            
            # Store listener reference for shutdown
            root_logger._queue_listener = queue_listener
        else:
            root_logger.addHandler(file_handler)

        root_logger.info(f"Logging initialized - Log file: {log_file}")

    # Set per-module log levels
    if module_levels:
        for module_name, level in module_levels.items():
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(level)
            root_logger.debug(f"Set log level for '{module_name}': {logging.getLevelName(level)}")

    return root_logger


def shutdown_logging():
    """Shutdown async logging gracefully."""
    root_logger = logging.getLogger()
    if hasattr(root_logger, '_queue_listener'):
        root_logger._queue_listener.stop()


def get_logger(name):
    """Dapatkan logger instance untuk modul tertentu."""
    return logging.getLogger(name)


class PerformanceLogger:
    """Context manager untuk logging metrik performa (time + memory)."""

    def __init__(self, logger, operation_name, track_memory=True):
        self.logger = logger
        self.operation_name = operation_name
        self.track_memory = track_memory
        self.start_time = None
        self.start_memory = None

    def __enter__(self):
        import time
        self.start_time = time.time()
        
        if self.track_memory:
            try:
                process = psutil.Process()
                self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
            except Exception:
                self.start_memory = None
        
        self.logger.debug(f"Started: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        elapsed = time.time() - self.start_time
        
        memory_delta = None
        if self.track_memory and self.start_memory is not None:
            try:
                process = psutil.Process()
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_delta = end_memory - self.start_memory
            except Exception:
                pass
        
        if exc_type is None:
            msg = f"Completed: {self.operation_name} | Time: {elapsed:.3f}s"
            if memory_delta is not None:
                msg += f" | Memory: {memory_delta:+.1f}MB"
            self.logger.info(msg)
        else:
            msg = f"Failed: {self.operation_name} | Time: {elapsed:.3f}s | Error: {exc_val}"
            if memory_delta is not None:
                msg += f" | Memory: {memory_delta:+.1f}MB"
            self.logger.error(msg)
        
        return False
