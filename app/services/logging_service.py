"""Structured logging service for production"""
import logging
import logging.handlers
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
            }

        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data

        # Add correlation ID if present
        if hasattr(record, "correlation_id"):
            log_data["correlation_id"] = record.correlation_id

        # Add user ID if present
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id

        # Add request ID if present
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        return json.dumps(log_data)


class LoggingService:
    """Centralized logging service with structured logging"""

    def __init__(
        self,
        app_name: str = "realtor-ai-copilot",
        log_level: str = "INFO",
        log_format: str = "json",
        log_file: Optional[str] = None,
        log_max_size_mb: int = 100,
        log_backup_count: int = 10
    ):
        """Initialize logging service

        Args:
            app_name: Application name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_format: Log format (json or text)
            log_file: Path to log file (None for stdout only)
            log_max_size_mb: Maximum size of log file before rotation
            log_backup_count: Number of backup log files to keep
        """
        self.app_name = app_name
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.log_format = log_format
        self.log_file = log_file

        # Configure root logger
        self.logger = logging.getLogger(app_name)
        self.logger.setLevel(self.log_level)
        self.logger.propagate = False

        # Remove existing handlers
        self.logger.handlers.clear()

        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)

        if log_format == "json":
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )

        self.logger.addHandler(console_handler)

        # Add file handler if specified
        if log_file:
            # Create log directory if it doesn't exist
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=log_max_size_mb * 1024 * 1024,
                backupCount=log_backup_count
            )
            file_handler.setLevel(self.log_level)

            if log_format == "json":
                file_handler.setFormatter(StructuredFormatter())
            else:
                file_handler.setFormatter(
                    logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                )

            self.logger.addHandler(file_handler)

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log debug message

        Args:
            message: Log message
            extra: Extra data to include
            **kwargs: Additional context (user_id, request_id, etc.)
        """
        self._log(logging.DEBUG, message, extra, **kwargs)

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log info message

        Args:
            message: Log message
            extra: Extra data to include
            **kwargs: Additional context (user_id, request_id, etc.)
        """
        self._log(logging.INFO, message, extra, **kwargs)

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log warning message

        Args:
            message: Log message
            extra: Extra data to include
            **kwargs: Additional context (user_id, request_id, etc.)
        """
        self._log(logging.WARNING, message, extra, **kwargs)

    def error(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
        **kwargs
    ):
        """Log error message

        Args:
            message: Log message
            extra: Extra data to include
            exc_info: Include exception information
            **kwargs: Additional context (user_id, request_id, etc.)
        """
        self._log(logging.ERROR, message, extra, exc_info=exc_info, **kwargs)

    def critical(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
        **kwargs
    ):
        """Log critical message

        Args:
            message: Log message
            extra: Extra data to include
            exc_info: Include exception information
            **kwargs: Additional context (user_id, request_id, etc.)
        """
        self._log(logging.CRITICAL, message, extra, exc_info=exc_info, **kwargs)

    def _log(
        self,
        level: int,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
        **kwargs
    ):
        """Internal log method

        Args:
            level: Log level
            message: Log message
            extra: Extra data to include
            exc_info: Include exception information
            **kwargs: Additional context
        """
        # Create log record with extra context
        record = logging.LogRecord(
            name=self.logger.name,
            level=level,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )

        # Add extra data
        if extra:
            record.extra_data = extra

        # Add context from kwargs
        for key, value in kwargs.items():
            setattr(record, key, value)

        # Handle exception info
        if exc_info:
            record.exc_info = sys.exc_info()

        self.logger.handle(record)

    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        **extra
    ):
        """Log HTTP request

        Args:
            method: HTTP method
            path: Request path
            status_code: Response status code
            duration_ms: Request duration in milliseconds
            user_id: User ID
            request_id: Request ID
            **extra: Additional data
        """
        self.info(
            f"{method} {path} - {status_code} ({duration_ms:.2f}ms)",
            extra={
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": duration_ms,
                **extra
            },
            user_id=user_id,
            request_id=request_id
        )

    def log_workflow_execution(
        self,
        workflow_id: str,
        execution_id: str,
        status: str,
        duration_ms: float,
        user_id: Optional[str] = None,
        **extra
    ):
        """Log workflow execution

        Args:
            workflow_id: Workflow identifier
            execution_id: Execution identifier
            status: Execution status
            duration_ms: Execution duration
            user_id: User ID
            **extra: Additional data
        """
        self.info(
            f"Workflow {workflow_id} execution {status}",
            extra={
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "status": status,
                "duration_ms": duration_ms,
                **extra
            },
            user_id=user_id
        )

    def log_ab_test_event(
        self,
        test_id: str,
        event_type: str,
        variant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **extra
    ):
        """Log A/B test event

        Args:
            test_id: Test identifier
            event_type: Event type (assigned, recorded, analyzed, etc.)
            variant_id: Variant identifier
            user_id: User ID
            **extra: Additional data
        """
        self.info(
            f"A/B test {event_type}: {test_id}",
            extra={
                "test_id": test_id,
                "event_type": event_type,
                "variant_id": variant_id,
                **extra
            },
            user_id=user_id
        )

    def log_cache_event(
        self,
        cache_name: str,
        event_type: str,
        key: Optional[str] = None,
        **extra
    ):
        """Log cache event

        Args:
            cache_name: Cache name
            event_type: Event type (hit, miss, set, clear, etc.)
            key: Cache key
            **extra: Additional data
        """
        self.debug(
            f"Cache {event_type}: {cache_name}",
            extra={
                "cache_name": cache_name,
                "event_type": event_type,
                "key": key,
                **extra
            }
        )

    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        **extra
    ):
        """Log security event

        Args:
            event_type: Event type (auth_failed, rate_limited, etc.)
            severity: Severity level
            description: Event description
            user_id: User ID
            ip_address: IP address
            **extra: Additional data
        """
        log_method = self.warning if severity == "high" else self.info

        log_method(
            f"Security event: {event_type} - {description}",
            extra={
                "event_type": event_type,
                "severity": severity,
                "description": description,
                "ip_address": ip_address,
                **extra
            },
            user_id=user_id
        )


# Global logging service instance
logging_service = LoggingService(
    app_name="realtor-ai-copilot",
    log_level="INFO",
    log_format="json",
    log_file="logs/app.log"
)
