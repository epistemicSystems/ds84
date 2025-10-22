"""Rate limiting middleware for API protection"""
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib


class RateLimiter:
    """Token bucket rate limiter"""

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        requests_per_day: int = 10000
    ):
        """Initialize rate limiter

        Args:
            requests_per_minute: Max requests per minute
            requests_per_hour: Max requests per hour
            requests_per_day: Max requests per day
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests_per_day = requests_per_day

        # Request history: {client_id: {window: [timestamps]}}
        self.request_history: Dict[str, Dict[str, list]] = defaultdict(
            lambda: {
                "minute": [],
                "hour": [],
                "day": []
            }
        )

    def check_rate_limit(
        self,
        client_id: str,
        endpoint: Optional[str] = None
    ) -> tuple[bool, Dict[str, any]]:
        """Check if request is within rate limits

        Args:
            client_id: Client identifier (IP, user_id, API key)
            endpoint: Optional endpoint-specific limit

        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        now = datetime.utcnow()

        # Clean old entries
        self._cleanup_old_entries(client_id, now)

        # Get current counts
        history = self.request_history[client_id]

        minute_count = len(history["minute"])
        hour_count = len(history["hour"])
        day_count = len(history["day"])

        # Check limits
        if minute_count >= self.requests_per_minute:
            return False, {
                "limit": self.requests_per_minute,
                "remaining": 0,
                "reset": self._get_reset_time("minute", history),
                "window": "minute"
            }

        if hour_count >= self.requests_per_hour:
            return False, {
                "limit": self.requests_per_hour,
                "remaining": 0,
                "reset": self._get_reset_time("hour", history),
                "window": "hour"
            }

        if day_count >= self.requests_per_day:
            return False, {
                "limit": self.requests_per_day,
                "remaining": 0,
                "reset": self._get_reset_time("day", history),
                "window": "day"
            }

        # Record request
        timestamp = now.timestamp()
        history["minute"].append(timestamp)
        history["hour"].append(timestamp)
        history["day"].append(timestamp)

        return True, {
            "limit": self.requests_per_minute,
            "remaining": self.requests_per_minute - (minute_count + 1),
            "reset": (now + timedelta(minutes=1)).isoformat(),
            "window": "minute"
        }

    def _cleanup_old_entries(self, client_id: str, now: datetime):
        """Remove old timestamp entries

        Args:
            client_id: Client identifier
            now: Current time
        """
        history = self.request_history[client_id]

        # Remove entries older than time windows
        cutoff_minute = (now - timedelta(minutes=1)).timestamp()
        cutoff_hour = (now - timedelta(hours=1)).timestamp()
        cutoff_day = (now - timedelta(days=1)).timestamp()

        history["minute"] = [t for t in history["minute"] if t > cutoff_minute]
        history["hour"] = [t for t in history["hour"] if t > cutoff_hour]
        history["day"] = [t for t in history["day"] if t > cutoff_day]

    def _get_reset_time(self, window: str, history: Dict[str, list]) -> str:
        """Get reset time for a window

        Args:
            window: Time window (minute, hour, day)
            history: Request history

        Returns:
            ISO timestamp of reset time
        """
        if not history[window]:
            return datetime.utcnow().isoformat()

        oldest_timestamp = min(history[window])
        oldest_time = datetime.fromtimestamp(oldest_timestamp)

        if window == "minute":
            reset_time = oldest_time + timedelta(minutes=1)
        elif window == "hour":
            reset_time = oldest_time + timedelta(hours=1)
        else:  # day
            reset_time = oldest_time + timedelta(days=1)

        return reset_time.isoformat()

    def get_client_id(self, request: Request) -> str:
        """Extract client identifier from request

        Args:
            request: FastAPI request

        Returns:
            Client identifier
        """
        # Try API key first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return hashlib.sha256(api_key.encode()).hexdigest()[:16]

        # Fall back to IP address
        # Check for forwarded IP (behind proxy)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        # Direct IP
        return request.client.host if request.client else "unknown"


# Global rate limiter
rate_limiter = RateLimiter(
    requests_per_minute=60,
    requests_per_hour=1000,
    requests_per_day=10000
)


async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware

    Args:
        request: FastAPI request
        call_next: Next middleware/endpoint

    Returns:
        Response or rate limit error
    """
    # Skip rate limiting for health check
    if request.url.path == "/health":
        return await call_next(request)

    # Get client ID
    client_id = rate_limiter.get_client_id(request)

    # Check rate limit
    allowed, rate_info = rate_limiter.check_rate_limit(client_id)

    if not allowed:
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "limit": rate_info["limit"],
                "window": rate_info["window"],
                "reset": rate_info["reset"],
                "message": f"Too many requests. Please try again after {rate_info['reset']}"
            },
            headers={
                "X-RateLimit-Limit": str(rate_info["limit"]),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": rate_info["reset"],
                "Retry-After": "60"
            }
        )

    # Process request
    response = await call_next(request)

    # Add rate limit headers
    response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
    response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
    response.headers["X-RateLimit-Reset"] = rate_info["reset"]

    return response
