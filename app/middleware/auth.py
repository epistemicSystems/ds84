"""API authentication and authorization middleware"""
from fastapi import Request, HTTPException, Security
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, List
import hashlib
import os
from datetime import datetime, timedelta


# API Key security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Bearer token security scheme
bearer_scheme = HTTPBearer(auto_error=False)


class APIKeyManager:
    """Manages API keys and permissions"""

    def __init__(self):
        """Initialize API key manager"""
        # In production, load from secure storage (database, secrets manager)
        self.api_keys = {
            # Format: hashed_key: {user_id, permissions, created_at, rate_limit}
        }

        # Load from environment for demo
        demo_key = os.getenv("API_KEY")
        if demo_key:
            self.api_keys[self._hash_key(demo_key)] = {
                "user_id": "admin",
                "permissions": ["*"],  # All permissions
                "created_at": datetime.utcnow(),
                "rate_limit_multiplier": 10.0  # 10x normal rate limit
            }

    def validate_api_key(self, api_key: str) -> Optional[dict]:
        """Validate API key

        Args:
            api_key: API key to validate

        Returns:
            Key info if valid, None otherwise
        """
        if not api_key:
            return None

        hashed = self._hash_key(api_key)

        return self.api_keys.get(hashed)

    def create_api_key(
        self,
        user_id: str,
        permissions: List[str],
        rate_limit_multiplier: float = 1.0
    ) -> str:
        """Create a new API key

        Args:
            user_id: User identifier
            permissions: List of permissions
            rate_limit_multiplier: Rate limit multiplier

        Returns:
            API key (unhashed)
        """
        # Generate random key
        import secrets
        api_key = f"rac_{secrets.token_urlsafe(32)}"

        # Store hashed version
        hashed = self._hash_key(api_key)
        self.api_keys[hashed] = {
            "user_id": user_id,
            "permissions": permissions,
            "created_at": datetime.utcnow(),
            "rate_limit_multiplier": rate_limit_multiplier
        }

        return api_key

    def revoke_api_key(self, api_key: str):
        """Revoke an API key

        Args:
            api_key: API key to revoke
        """
        hashed = self._hash_key(api_key)
        if hashed in self.api_keys:
            del self.api_keys[hashed]

    def check_permission(
        self,
        api_key_info: dict,
        required_permission: str
    ) -> bool:
        """Check if API key has required permission

        Args:
            api_key_info: API key information
            required_permission: Required permission

        Returns:
            True if authorized
        """
        permissions = api_key_info.get("permissions", [])

        # Wildcard permission
        if "*" in permissions:
            return True

        # Exact permission match
        if required_permission in permissions:
            return True

        # Prefix match (e.g., "workflow.*" matches "workflow.execute")
        for perm in permissions:
            if perm.endswith(".*"):
                prefix = perm[:-2]
                if required_permission.startswith(prefix):
                    return True

        return False

    def _hash_key(self, api_key: str) -> str:
        """Hash API key for storage

        Args:
            api_key: API key

        Returns:
            Hashed key
        """
        return hashlib.sha256(api_key.encode()).hexdigest()


# Global API key manager
api_key_manager = APIKeyManager()


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header)
) -> dict:
    """Verify API key from header

    Args:
        api_key: API key from header

    Returns:
        API key info

    Raises:
        HTTPException if invalid
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Please provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    key_info = api_key_manager.validate_api_key(api_key)

    if not key_info:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    return key_info


async def verify_bearer_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme)
) -> dict:
    """Verify bearer token from header

    Args:
        credentials: Bearer token credentials

    Returns:
        Token info

    Raises:
        HTTPException if invalid
    """
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Bearer token required",
            headers={"WWW-Authenticate": "Bearer"}
        )

    token = credentials.credentials

    # In production, validate JWT token
    # For demo, use simple validation
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    return {"user_id": "authenticated_user", "permissions": ["read"]}


def require_permission(permission: str):
    """Decorator to require specific permission

    Args:
        permission: Required permission

    Returns:
        Dependency function
    """
    async def permission_checker(
        api_key_info: dict = Security(verify_api_key)
    ) -> dict:
        if not api_key_manager.check_permission(api_key_info, permission):
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied. Required: {permission}"
            )
        return api_key_info

    return permission_checker


# Public endpoints that don't require authentication
PUBLIC_ENDPOINTS = [
    "/",
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json"
]


async def auth_middleware(request: Request, call_next):
    """Authentication middleware

    Args:
        request: FastAPI request
        call_next: Next middleware/endpoint

    Returns:
        Response or auth error
    """
    # Skip auth for public endpoints
    if request.url.path in PUBLIC_ENDPOINTS:
        return await call_next(request)

    # Check for API key
    api_key = request.headers.get("X-API-Key")

    if api_key:
        key_info = api_key_manager.validate_api_key(api_key)
        if not key_info:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid API key"}
            )

        # Add user info to request state
        request.state.user_id = key_info.get("user_id")
        request.state.permissions = key_info.get("permissions", [])

        return await call_next(request)

    # For development, allow unauthenticated access
    if os.getenv("ENVIRONMENT") == "development":
        request.state.user_id = "dev_user"
        request.state.permissions = ["*"]
        return await call_next(request)

    # Require authentication in production
    return JSONResponse(
        status_code=401,
        content={
            "error": "Authentication required",
            "message": "Please provide X-API-Key header or set ENVIRONMENT=development"
        }
    )
