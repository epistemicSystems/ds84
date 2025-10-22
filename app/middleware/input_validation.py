"""Input validation and sanitization middleware for security"""
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import re
import json


class InputValidator:
    """Validates and sanitizes user input to prevent common attacks"""

    def __init__(
        self,
        max_request_size_mb: float = 10.0,
        max_string_length: int = 100000,
        enable_sql_injection_check: bool = True,
        enable_xss_check: bool = True,
        enable_command_injection_check: bool = True
    ):
        """Initialize input validator

        Args:
            max_request_size_mb: Maximum request body size in MB
            max_string_length: Maximum length for string fields
            enable_sql_injection_check: Enable SQL injection detection
            enable_xss_check: Enable XSS detection
            enable_command_injection_check: Enable command injection detection
        """
        self.max_request_size_bytes = int(max_request_size_mb * 1024 * 1024)
        self.max_string_length = max_string_length
        self.enable_sql_injection_check = enable_sql_injection_check
        self.enable_xss_check = enable_xss_check
        self.enable_command_injection_check = enable_command_injection_check

        # SQL injection patterns
        self.sql_injection_patterns = [
            r"(\bUNION\b.*\bSELECT\b)",
            r"(\bINSERT\b.*\bINTO\b)",
            r"(\bDELETE\b.*\bFROM\b)",
            r"(\bDROP\b.*\bTABLE\b)",
            r"(\bUPDATE\b.*\bSET\b)",
            r"(--\s*$)",
            r"(/\*.*\*/)",
            r"(\bOR\b.*=.*)",
            r"(\bAND\b.*=.*)",
            r"(';.*--)",
            r"(1=1)",
            r"(1' OR '1)",
        ]

        # XSS patterns
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",  # Event handlers like onclick=
            r"<iframe",
            r"<embed",
            r"<object",
            r"eval\s*\(",
            r"expression\s*\(",
        ]

        # Command injection patterns
        self.command_injection_patterns = [
            r";\s*(rm|cat|ls|wget|curl|nc|bash|sh|python|perl|ruby)",
            r"\|\s*(rm|cat|ls|wget|curl|nc|bash|sh)",
            r"`.*`",
            r"\$\(.*\)",
            r"&&\s*(rm|cat|ls|wget|curl)",
        ]

        # Compile patterns for performance
        self.sql_injection_regex = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.sql_injection_patterns
        ]
        self.xss_regex = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.xss_patterns
        ]
        self.command_injection_regex = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.command_injection_patterns
        ]

    def check_sql_injection(self, value: str) -> bool:
        """Check if string contains SQL injection patterns

        Args:
            value: String to check

        Returns:
            True if suspicious pattern found
        """
        if not self.enable_sql_injection_check:
            return False

        for pattern in self.sql_injection_regex:
            if pattern.search(value):
                return True

        return False

    def check_xss(self, value: str) -> bool:
        """Check if string contains XSS patterns

        Args:
            value: String to check

        Returns:
            True if suspicious pattern found
        """
        if not self.enable_xss_check:
            return False

        for pattern in self.xss_regex:
            if pattern.search(value):
                return True

        return False

    def check_command_injection(self, value: str) -> bool:
        """Check if string contains command injection patterns

        Args:
            value: String to check

        Returns:
            True if suspicious pattern found
        """
        if not self.enable_command_injection_check:
            return False

        for pattern in self.command_injection_regex:
            if pattern.search(value):
                return True

        return False

    def sanitize_string(self, value: str) -> str:
        """Sanitize string by removing potentially dangerous characters

        Args:
            value: String to sanitize

        Returns:
            Sanitized string
        """
        # Trim to max length
        if len(value) > self.max_string_length:
            value = value[:self.max_string_length]

        # Remove null bytes
        value = value.replace("\x00", "")

        # Normalize whitespace
        value = " ".join(value.split())

        return value

    def validate_string(self, value: str, field_name: str = "field") -> str:
        """Validate string field

        Args:
            value: String to validate
            field_name: Field name for error messages

        Returns:
            Sanitized string

        Raises:
            HTTPException if validation fails
        """
        # Check length
        if len(value) > self.max_string_length:
            raise HTTPException(
                status_code=400,
                detail=f"{field_name} exceeds maximum length of {self.max_string_length}"
            )

        # Check for attacks
        if self.check_sql_injection(value):
            raise HTTPException(
                status_code=400,
                detail=f"{field_name} contains potentially malicious SQL patterns"
            )

        if self.check_xss(value):
            raise HTTPException(
                status_code=400,
                detail=f"{field_name} contains potentially malicious script patterns"
            )

        if self.check_command_injection(value):
            raise HTTPException(
                status_code=400,
                detail=f"{field_name} contains potentially malicious command patterns"
            )

        # Sanitize
        return self.sanitize_string(value)

    def validate_dict(
        self,
        data: Dict[str, Any],
        depth: int = 0,
        max_depth: int = 10
    ) -> Dict[str, Any]:
        """Recursively validate dictionary

        Args:
            data: Dictionary to validate
            depth: Current recursion depth
            max_depth: Maximum recursion depth

        Returns:
            Validated dictionary

        Raises:
            HTTPException if validation fails
        """
        if depth > max_depth:
            raise HTTPException(
                status_code=400,
                detail="Request data exceeds maximum nesting depth"
            )

        validated = {}

        for key, value in data.items():
            # Validate key
            if not isinstance(key, str):
                key = str(key)

            key = self.validate_string(key, f"key '{key}'")

            # Validate value based on type
            if isinstance(value, str):
                validated[key] = self.validate_string(value, f"field '{key}'")
            elif isinstance(value, dict):
                validated[key] = self.validate_dict(value, depth + 1, max_depth)
            elif isinstance(value, list):
                validated[key] = self.validate_list(value, depth + 1, max_depth)
            else:
                # Numbers, booleans, None pass through
                validated[key] = value

        return validated

    def validate_list(
        self,
        data: list,
        depth: int = 0,
        max_depth: int = 10
    ) -> list:
        """Recursively validate list

        Args:
            data: List to validate
            depth: Current recursion depth
            max_depth: Maximum recursion depth

        Returns:
            Validated list

        Raises:
            HTTPException if validation fails
        """
        if depth > max_depth:
            raise HTTPException(
                status_code=400,
                detail="Request data exceeds maximum nesting depth"
            )

        validated = []

        for value in data:
            if isinstance(value, str):
                validated.append(self.validate_string(value, "list item"))
            elif isinstance(value, dict):
                validated.append(self.validate_dict(value, depth + 1, max_depth))
            elif isinstance(value, list):
                validated.append(self.validate_list(value, depth + 1, max_depth))
            else:
                validated.append(value)

        return validated


# Global validator instance
input_validator = InputValidator()


async def input_validation_middleware(request: Request, call_next):
    """Input validation middleware

    Validates and sanitizes request data to prevent common attacks.

    Args:
        request: FastAPI request
        call_next: Next middleware/endpoint

    Returns:
        Response or validation error
    """
    # Skip validation for public endpoints
    public_endpoints = ["/", "/health", "/docs", "/redoc", "/openapi.json"]

    if request.url.path in public_endpoints:
        return await call_next(request)

    # Check request size
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            content_length_int = int(content_length)
            if content_length_int > input_validator.max_request_size_bytes:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "Request too large",
                        "max_size_mb": input_validator.max_request_size_bytes / (1024 * 1024)
                    }
                )
        except ValueError:
            pass

    # For POST/PUT/PATCH requests with JSON body, validate the body
    if request.method in ["POST", "PUT", "PATCH"]:
        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type:
            try:
                # Read and parse body
                body = await request.body()

                if not body:
                    return await call_next(request)

                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Invalid JSON"}
                    )

                # Validate data
                if isinstance(data, dict):
                    validated_data = input_validator.validate_dict(data)
                elif isinstance(data, list):
                    validated_data = input_validator.validate_list(data)
                else:
                    # Primitive types
                    validated_data = data

                # Replace request body with validated data
                # Note: This is a simplified approach. In production, you might want
                # to use a custom Request class to properly handle body replacement
                request._body = json.dumps(validated_data).encode()

            except HTTPException as e:
                return JSONResponse(
                    status_code=e.status_code,
                    content={"error": e.detail}
                )
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"error": "Validation error", "detail": str(e)}
                )

    # Validate query parameters
    try:
        for key, value in request.query_params.items():
            # Validate query parameter
            input_validator.validate_string(value, f"query parameter '{key}'")
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"error": e.detail}
        )

    # Validate path parameters
    try:
        for key, value in request.path_params.items():
            if isinstance(value, str):
                input_validator.validate_string(value, f"path parameter '{key}'")
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"error": e.detail}
        )

    # Continue to next middleware/endpoint
    response = await call_next(request)
    return response
