# Week 6: Advanced Personalization & Refinement

## Overview

Week 6 focuses on production-ready system hardening, including A/B testing, rate limiting, authentication, input validation, and performance optimization through caching. These features ensure the system is secure, scalable, and optimized for production deployment.

---

## Table of Contents

1. [A/B Testing Framework](#ab-testing-framework)
2. [Rate Limiting](#rate-limiting)
3. [Authentication & Authorization](#authentication--authorization)
4. [Input Validation & Security](#input-validation--security)
5. [Caching for Performance](#caching-for-performance)
6. [API Endpoints](#api-endpoints)
7. [Testing](#testing)
8. [Configuration](#configuration)

---

## A/B Testing Framework

### Overview

The A/B testing framework enables data-driven optimization of workflows, prompts, models, and routing strategies through controlled experimentation with statistical significance testing.

### Key Features

- **Multiple Test Types**: Support for workflow, prompt, model, and routing tests
- **Hash-Based Assignment**: Consistent variant assignment for users
- **Statistical Analysis**: Confidence intervals, p-values, winner determination
- **Traffic Allocation**: Flexible percentage-based traffic distribution
- **Metrics Tracking**: Multi-metric analysis with customizable primary metric

### Creating a Test

```python
from app.services.ab_testing_framework import ab_testing_framework

test_id = ab_testing_framework.create_test(
    test_name="GPT-4 vs GPT-3.5 Comparison",
    test_type="model",
    variants=[
        {
            "variant_id": "gpt35",
            "variant_name": "GPT-3.5 Turbo",
            "configuration": {"model": "gpt-3.5-turbo"},
            "traffic_percentage": 50.0
        },
        {
            "variant_id": "gpt4",
            "variant_name": "GPT-4",
            "configuration": {"model": "gpt-4"},
            "traffic_percentage": 50.0
        }
    ],
    control_variant_id="gpt35",
    primary_metric="success_rate",
    min_sample_size=100
)

# Start test
ab_testing_framework.start_test(test_id)
```

### Variant Assignment

```python
# Assign user to variant (consistent hashing)
variant_id = ab_testing_framework.assign_variant(test_id, user_id)

# Use variant configuration
test = ab_testing_framework.get_test(test_id)
variant = next(v for v in test.variants if v.variant_id == variant_id)
config = variant.configuration
```

### Recording Results

```python
ab_testing_framework.record_result(
    test_id=test_id,
    user_id=user_id,
    variant_id=variant_id,
    metrics={
        "success_rate": 0.85,
        "latency": 1.2,
        "cost": 0.003,
        "quality_score": 4.5
    }
)
```

### Analyzing Results

```python
result = ab_testing_framework.analyze_test(test_id)

print(f"Winner: {result.winner_variant_id}")
print(f"Confidence: {result.confidence_level}")
print(f"P-value: {result.p_value}")

for variant_result in result.variant_results:
    print(f"{variant_result.variant_name}:")
    print(f"  Mean: {variant_result.mean_value}")
    print(f"  CI: [{variant_result.confidence_interval_lower}, "
          f"{variant_result.confidence_interval_upper}]")
```

### API Endpoints

```bash
# Create test
POST /api/ab-testing/tests

# List tests
GET /api/ab-testing/tests?status=running&test_type=workflow

# Get test details
GET /api/ab-testing/tests/{test_id}

# Assign variant
POST /api/ab-testing/tests/{test_id}/assign

# Record result
POST /api/ab-testing/tests/{test_id}/record

# Analyze test
GET /api/ab-testing/tests/{test_id}/analyze

# Start/Stop test
POST /api/ab-testing/tests/{test_id}/start
POST /api/ab-testing/tests/{test_id}/stop
```

---

## Rate Limiting

### Overview

Token bucket rate limiting protects the API from abuse with multiple time windows (minute, hour, day) and endpoint-specific limits.

### Features

- **Multi-Window Limiting**: Per-minute, per-hour, and per-day limits
- **Client Identification**: Based on API key or IP address
- **Endpoint-Specific Limits**: Different limits for expensive operations
- **Rate Limit Headers**: Standard X-RateLimit-* headers
- **Automatic Cleanup**: Old request history is cleaned automatically

### Configuration

```python
from app.middleware.rate_limiter import RateLimiter

limiter = RateLimiter(
    requests_per_minute=60,
    requests_per_hour=1000,
    requests_per_day=10000,
    endpoint_limits={
        "/api/metacognitive/self-improve": {
            "minute": 5,
            "hour": 20,
            "day": 100
        }
    }
)
```

### Rate Limit Information

Response headers include:
- `X-RateLimit-Limit-Minute`: Requests allowed per minute
- `X-RateLimit-Remaining-Minute`: Remaining requests in current minute
- `X-RateLimit-Reset-Minute`: Timestamp when limit resets
- Similar headers for `Hour` and `Day` windows

### Handling Rate Limits

When rate limited (429 status):
```json
{
  "error": "Rate limit exceeded",
  "retry_after": 45,
  "limits": {
    "minute": {"limit": 60, "remaining": 0, "reset_at": "2025-10-22T10:31:00Z"},
    "hour": {"limit": 1000, "remaining": 823, "reset_at": "2025-10-22T11:00:00Z"}
  }
}
```

### API Key Rate Multipliers

API keys can have rate limit multipliers:
```python
api_key = api_key_manager.create_api_key(
    user_id="premium_user",
    permissions=["*"],
    rate_limit_multiplier=10.0  # 10x normal limits
)
```

---

## Authentication & Authorization

### Overview

API key-based authentication with granular permission control and secure key storage using SHA-256 hashing.

### Creating API Keys

```python
from app.middleware.auth import api_key_manager

api_key = api_key_manager.create_api_key(
    user_id="david",
    permissions=[
        "workflow.*",        # All workflow operations
        "query.search",      # Search queries
        "feedback.submit"    # Submit feedback
    ],
    rate_limit_multiplier=5.0
)

# Returns: "rac_Xj8kP4mN2qR7vT9yU3bD5fH6gK8lM0nP1sQ4wE7"
```

### Using API Keys

```bash
curl -H "X-API-Key: rac_Xj8kP4mN2qR7vT9yU3bD5fH6gK8lM0nP1sQ4wE7" \
     http://localhost:8000/api/property-search \
     -d '{"query": "Find me a house"}'
```

### Permission System

Permissions support:
- **Exact matching**: `query.search` matches `query.search`
- **Wildcard matching**: `workflow.*` matches `workflow.execute`, `workflow.list`, etc.
- **Admin wildcard**: `*` grants all permissions

### Protecting Endpoints

```python
from fastapi import Security
from app.middleware.auth import require_permission

@app.post("/api/admin/clear-cache")
async def clear_cache(
    api_key_info: dict = Security(require_permission("admin.cache"))
):
    # Only accessible with admin.cache or admin.* or * permission
    cache_service.clear_all()
    return {"status": "cleared"}
```

### Development Mode

For development, set `ENVIRONMENT=development` in `.env` to bypass authentication:
```bash
ENVIRONMENT=development
```

### Revoking API Keys

```python
api_key_manager.revoke_api_key(api_key)
```

---

## Input Validation & Security

### Overview

Comprehensive input validation middleware protects against SQL injection, XSS, command injection, and other common attacks.

### Security Features

- **SQL Injection Detection**: Pattern matching for SQL attacks
- **XSS Protection**: Detects malicious scripts and event handlers
- **Command Injection Detection**: Prevents shell command execution
- **Request Size Limiting**: Maximum 10MB requests by default
- **String Length Limiting**: Prevents memory exhaustion
- **Null Byte Filtering**: Removes null bytes from input
- **Recursive Validation**: Validates nested dictionaries and lists

### Attack Detection Examples

#### SQL Injection
```python
# These will be rejected:
"SELECT * FROM users WHERE 1=1"
"admin' OR '1'='1"
"DROP TABLE users; --"
```

#### XSS
```python
# These will be rejected:
"<script>alert('xss')</script>"
"<img src=x onerror=alert(1)>"
"javascript:alert(1)"
```

#### Command Injection
```python
# These will be rejected:
"; rm -rf /"
"| cat /etc/passwd"
"$(whoami)"
```

### Validation Configuration

```python
from app.middleware.input_validation import InputValidator

validator = InputValidator(
    max_request_size_mb=10.0,
    max_string_length=100000,
    enable_sql_injection_check=True,
    enable_xss_check=True,
    enable_command_injection_check=True
)
```

### Manual Validation

```python
from app.middleware.input_validation import input_validator

# Validate string
safe_query = input_validator.validate_string(
    user_input,
    field_name="query"
)

# Validate dictionary
safe_data = input_validator.validate_dict(request_data)
```

---

## Caching for Performance

### Overview

Multi-layer LRU caching with TTL support reduces API costs and improves response times for embeddings, prompts, and queries.

### Cache Layers

1. **Embedding Cache**: 24-hour TTL, 10,000 entries
2. **Prompt Cache**: 1-hour TTL, 1,000 entries
3. **Query Cache**: 30-minute TTL, 5,000 entries
4. **General Cache**: Configurable TTL, 1,000 entries

### Caching Embeddings

```python
from app.services.cache_service import cache_service

# Cache embedding
text = "Modern 3-bedroom house with pool"
embedding = [0.1, 0.2, 0.3, ...]  # From OpenAI API

cache_service.cache_embedding(text, embedding)

# Retrieve embedding
cached_embedding = cache_service.get_embedding(text)

if cached_embedding is None:
    # Generate new embedding
    embedding = await embedding_service.get_embedding(text)
    cache_service.cache_embedding(text, embedding)
else:
    embedding = cached_embedding
```

### Caching Prompts

```python
# Cache prompt result
prompt_key = "property_search.intent_analysis"
variables = {"query": "Find me a house"}
result = "Intent: property_search..."

cache_service.cache_prompt_result(prompt_key, variables, result, ttl_seconds=3600)

# Retrieve
cached_result = cache_service.get_prompt_result(prompt_key, variables)
```

### Caching Queries

```python
# Cache query result
query = "3-bedroom house in San Jose"
user_id = "user_123"
result = {
    "properties": [...],
    "count": 10
}

cache_service.cache_query_result(query, user_id, result, ttl_seconds=1800)

# Retrieve
cached_result = cache_service.get_query_result(query, user_id)
```

### Decorator-Based Caching

```python
from app.services.cache_service import cache_service

@cache_service.cached("expensive_computation", ttl_seconds=3600)
def expensive_function(param1, param2):
    # Expensive operation
    result = complex_calculation(param1, param2)
    return result

# First call executes function
result = expensive_function(1, 2)

# Second call uses cache
result = expensive_function(1, 2)  # Instant return
```

### Cache Statistics

```python
stats = cache_service.get_stats()

print(f"Embedding Cache:")
print(f"  Entries: {stats['embedding_cache']['entries']}")
print(f"  Hit Rate: {stats['embedding_cache']['hit_rate']:.2%}")
print(f"  Size: {stats['embedding_cache']['total_size_mb']:.2f} MB")
```

### Cache Invalidation

```python
# Clear specific category
cache_service.invalidate_category("property_search")

# Clear all caches
cache_service.clear_all()
```

### LRU Eviction

Caches use LRU (Least Recently Used) eviction:
- When cache reaches max size, oldest unused entry is evicted
- Accessing an entry marks it as recently used
- Expired entries are cleaned periodically

---

## API Endpoints

### A/B Testing Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/ab-testing/tests` | Create new A/B test |
| GET | `/api/ab-testing/tests` | List all tests (with filters) |
| GET | `/api/ab-testing/tests/{test_id}` | Get test details |
| POST | `/api/ab-testing/tests/{test_id}/assign` | Assign user to variant |
| POST | `/api/ab-testing/tests/{test_id}/record` | Record test result |
| GET | `/api/ab-testing/tests/{test_id}/analyze` | Analyze test with statistics |
| POST | `/api/ab-testing/tests/{test_id}/start` | Start test |
| POST | `/api/ab-testing/tests/{test_id}/stop` | Stop test |

### Example: Creating and Running a Test

```bash
# 1. Create test
curl -X POST http://localhost:8000/api/ab-testing/tests \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "test_name": "Workflow Optimization",
    "test_type": "workflow",
    "variants": [
      {
        "variant_id": "control",
        "variant_name": "Current",
        "configuration": {},
        "traffic_percentage": 50.0
      },
      {
        "variant_id": "optimized",
        "variant_name": "Optimized",
        "configuration": {},
        "traffic_percentage": 50.0
      }
    ],
    "control_variant_id": "control",
    "primary_metric": "success_rate"
  }'

# 2. Start test
curl -X POST http://localhost:8000/api/ab-testing/tests/{test_id}/start \
  -H "X-API-Key: your_api_key"

# 3. Assign users and record results
curl -X POST http://localhost:8000/api/ab-testing/tests/{test_id}/record \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "user_id": "user_123",
    "variant_id": "optimized",
    "metrics": {
      "success_rate": 0.85,
      "latency": 1.2
    }
  }'

# 4. Analyze results
curl http://localhost:8000/api/ab-testing/tests/{test_id}/analyze \
  -H "X-API-Key: your_api_key"
```

---

## Testing

### Running Tests

```bash
# Run all Week 6 tests
pytest tests/test_week6.py -v

# Run specific test class
pytest tests/test_week6.py::TestABTestingFramework -v

# Run with coverage
pytest tests/test_week6.py --cov=app --cov-report=html
```

### Test Coverage

Week 6 test suite includes:
- **A/B Testing**: Test creation, variant assignment, result recording, statistical analysis
- **Caching**: LRU eviction, TTL expiration, hit/miss statistics
- **Rate Limiting**: Multi-window limits, endpoint-specific limits, reset behavior
- **Input Validation**: SQL injection, XSS, command injection detection
- **Authentication**: API key creation, validation, permission checking

---

## Configuration

### Environment Variables

```bash
# .env file

# Environment mode
ENVIRONMENT=development  # or production

# API Keys
API_KEY=your_api_key_here

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_REQUESTS_PER_HOUR=1000
RATE_LIMIT_REQUESTS_PER_DAY=10000

# Caching
CACHE_EMBEDDING_SIZE=10000
CACHE_PROMPT_SIZE=1000
CACHE_QUERY_SIZE=5000
CACHE_DEFAULT_TTL=3600

# Input Validation
INPUT_MAX_REQUEST_SIZE_MB=10.0
INPUT_MAX_STRING_LENGTH=100000
```

### Middleware Order

Middleware is applied in this order (important for security):
1. CORS
2. Input Validation (reject malicious input early)
3. Rate Limiting (prevent abuse)
4. Authentication (verify identity)

### Production Considerations

For production deployment:

1. **Authentication**: Always require API keys (`ENVIRONMENT=production`)
2. **Rate Limiting**: Adjust limits based on usage patterns
3. **Caching**: Monitor cache hit rates and adjust sizes
4. **Input Validation**: Keep patterns updated with new attack vectors
5. **A/B Testing**: Use adequate sample sizes (100+ per variant)

---

## Performance Impact

### Caching Benefits

With typical usage patterns:
- **Embedding Cache**: 80%+ hit rate → 80% reduction in embedding API calls
- **Query Cache**: 30-50% hit rate → 30-50% faster response times
- **Prompt Cache**: 20-40% hit rate → Cost savings on repeated queries

### Overhead

Middleware overhead (per request):
- Input Validation: <1ms
- Rate Limiting: <0.5ms
- Authentication: <0.5ms
- Total: <2ms additional latency

### Cost Savings Example

For 10,000 daily queries:
- Without caching: 10,000 embedding calls × $0.0001 = $1.00/day
- With 80% cache hit rate: 2,000 calls × $0.0001 = $0.20/day
- **Savings**: $0.80/day = $292/year

---

## Best Practices

### A/B Testing

1. **Sample Size**: Ensure min_sample_size is adequate (100+ per variant)
2. **Test Duration**: Run tests long enough to account for day/week variations
3. **Single Metric**: Focus on one primary metric for decision-making
4. **Avoid Peeking**: Let tests run to completion before analyzing

### Caching

1. **TTL Selection**: Balance freshness vs. hit rate
2. **Cache Warming**: Pre-populate caches for common queries
3. **Invalidation**: Clear caches when underlying data changes
4. **Monitoring**: Track hit rates and adjust sizes

### Security

1. **API Keys**: Rotate keys regularly
2. **Permissions**: Follow principle of least privilege
3. **Rate Limits**: Set conservative limits, increase as needed
4. **Input Validation**: Validate all user input, trust nothing

### Rate Limiting

1. **Graceful Degradation**: Handle rate limits gracefully in clients
2. **Retry Logic**: Implement exponential backoff
3. **Burst Allowance**: Consider short-term bursts in limit design
4. **Monitoring**: Track rate limit violations to detect abuse

---

## Troubleshooting

### Cache Not Working

```python
# Check cache stats
stats = cache_service.get_stats()
print(stats)

# Verify TTL not too short
cache_service.prompt_cache.default_ttl_seconds = 3600

# Clear and repopulate
cache_service.clear_all()
```

### Rate Limit Issues

```python
# Check current limits
limiter = RateLimiter()
allowed, info = limiter.check_rate_limit(client_id)
print(info)

# Increase limits temporarily
limiter.requests_per_minute = 120
```

### Authentication Failures

```python
# Verify API key
key_info = api_key_manager.validate_api_key(api_key)
if not key_info:
    print("Invalid API key")

# Check permissions
has_perm = api_key_manager.check_permission(key_info, "workflow.execute")
print(f"Has permission: {has_perm}")
```

---

## Summary

Week 6 delivers production-ready features:

✅ **A/B Testing**: Data-driven optimization with statistical rigor
✅ **Rate Limiting**: Protection against abuse and cost overruns
✅ **Authentication**: Secure API access with granular permissions
✅ **Input Validation**: Defense against common security attacks
✅ **Caching**: Performance optimization and cost reduction

The system is now ready for production deployment with enterprise-grade security, performance, and experimentation capabilities.
