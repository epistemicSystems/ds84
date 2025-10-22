"""Comprehensive test suite for Week 6: Advanced Personalization & Refinement"""
import pytest
import time
from datetime import datetime
from app.services.ab_testing_framework import ab_testing_framework, ABTest, Variant
from app.services.cache_service import cache_service, LRUCache
from app.middleware.rate_limiter import RateLimiter
from app.middleware.input_validation import InputValidator
from app.middleware.auth import APIKeyManager


class TestABTestingFramework:
    """Tests for A/B testing framework"""

    def test_create_test(self):
        """Test creating an A/B test"""
        test_id = ab_testing_framework.create_test(
            test_name="Property Search Optimization",
            test_type="workflow",
            variants=[
                {
                    "variant_id": "control",
                    "variant_name": "Current Workflow",
                    "configuration": {"model": "gpt-3.5-turbo"},
                    "traffic_percentage": 50.0
                },
                {
                    "variant_id": "optimized",
                    "variant_name": "GPT-4 Workflow",
                    "configuration": {"model": "gpt-4"},
                    "traffic_percentage": 50.0
                }
            ],
            control_variant_id="control",
            primary_metric="success_rate"
        )

        assert test_id is not None
        test = ab_testing_framework.get_test(test_id)
        assert test is not None
        assert test.test_name == "Property Search Optimization"
        assert len(test.variants) == 2

    def test_variant_assignment_consistency(self):
        """Test that variant assignment is consistent for same user"""
        test_id = ab_testing_framework.create_test(
            test_name="Consistency Test",
            test_type="workflow",
            variants=[
                {"variant_id": "a", "variant_name": "A", "configuration": {}, "traffic_percentage": 50.0},
                {"variant_id": "b", "variant_name": "B", "configuration": {}, "traffic_percentage": 50.0}
            ],
            control_variant_id="a",
            primary_metric="success_rate"
        )

        ab_testing_framework.start_test(test_id)

        # Same user should get same variant
        user_id = "test_user_123"
        variant1 = ab_testing_framework.assign_variant(test_id, user_id)
        variant2 = ab_testing_framework.assign_variant(test_id, user_id)
        variant3 = ab_testing_framework.assign_variant(test_id, user_id)

        assert variant1 == variant2 == variant3

    def test_record_and_analyze(self):
        """Test recording results and analyzing"""
        test_id = ab_testing_framework.create_test(
            test_name="Analysis Test",
            test_type="prompt",
            variants=[
                {"variant_id": "control", "variant_name": "Control", "configuration": {}, "traffic_percentage": 50.0},
                {"variant_id": "test", "variant_name": "Test", "configuration": {}, "traffic_percentage": 50.0}
            ],
            control_variant_id="control",
            primary_metric="success_rate",
            min_sample_size=5
        )

        ab_testing_framework.start_test(test_id)

        # Record results for control (lower success rate)
        for i in range(10):
            ab_testing_framework.record_result(
                test_id=test_id,
                user_id=f"user_{i}",
                variant_id="control",
                metrics={"success_rate": 0.5 + (i % 2) * 0.1}  # 0.5-0.6
            )

        # Record results for test (higher success rate)
        for i in range(10):
            ab_testing_framework.record_result(
                test_id=test_id,
                user_id=f"user_{i+10}",
                variant_id="test",
                metrics={"success_rate": 0.8 + (i % 2) * 0.1}  # 0.8-0.9
            )

        # Analyze
        result = ab_testing_framework.analyze_test(test_id)

        assert result is not None
        assert len(result.variant_results) == 2

        # Find test variant result
        test_result = next(vr for vr in result.variant_results if vr.variant_id == "test")
        assert test_result.mean_value > 0.7  # Should be around 0.85

    def test_list_tests(self):
        """Test listing tests with filters"""
        # Create a test
        test_id = ab_testing_framework.create_test(
            test_name="List Test",
            test_type="model",
            variants=[
                {"variant_id": "a", "variant_name": "A", "configuration": {}, "traffic_percentage": 100.0}
            ],
            control_variant_id="a",
            primary_metric="latency"
        )

        ab_testing_framework.start_test(test_id)

        # List running tests
        running_tests = ab_testing_framework.list_tests(status="running")
        assert len([t for t in running_tests if t.test_id == test_id]) > 0


class TestCacheService:
    """Tests for caching service"""

    def test_lru_cache_basic_operations(self):
        """Test basic LRU cache operations"""
        cache = LRUCache(max_size=3, default_ttl_seconds=60)

        # Set values
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Get values
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

        # Non-existent key
        assert cache.get("key4") is None

    def test_lru_eviction(self):
        """Test LRU eviction policy"""
        cache = LRUCache(max_size=3, default_ttl_seconds=60)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 to make it recently used
        cache.get("key1")

        # Add key4, should evict key2 (least recently used)
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"  # Still there
        assert cache.get("key2") is None      # Evicted
        assert cache.get("key3") == "value3"  # Still there
        assert cache.get("key4") == "value4"  # Newly added

    def test_cache_expiration(self):
        """Test cache TTL expiration"""
        cache = LRUCache(max_size=10, default_ttl_seconds=1)

        cache.set("key1", "value1", ttl_seconds=1)

        # Should exist immediately
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert cache.get("key1") is None

    def test_cache_stats(self):
        """Test cache statistics"""
        cache = LRUCache(max_size=10, default_ttl_seconds=60)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Hit
        cache.get("key1")
        # Miss
        cache.get("key3")

        stats = cache.get_stats()

        assert stats["entries"] == 2
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_embedding_cache(self):
        """Test embedding caching"""
        embedding = [0.1, 0.2, 0.3, 0.4]
        text = "test text"

        # Cache embedding
        cache_service.cache_embedding(text, embedding)

        # Retrieve embedding
        cached = cache_service.get_embedding(text)

        assert cached == embedding

    def test_query_cache(self):
        """Test query result caching"""
        query = "Find me a 3-bedroom house"
        user_id = "user_123"
        result = {
            "properties": [{"id": 1, "address": "123 Main St"}],
            "count": 1
        }

        # Cache query result
        cache_service.cache_query_result(query, user_id, result)

        # Retrieve
        cached = cache_service.get_query_result(query, user_id)

        assert cached == result

    def test_prompt_cache(self):
        """Test prompt result caching"""
        prompt_key = "property_search.intent_analysis"
        variables = {"query": "test query"}
        result = "Intent: property search"

        # Cache prompt result
        cache_service.cache_prompt_result(prompt_key, variables, result)

        # Retrieve
        cached = cache_service.get_prompt_result(prompt_key, variables)

        assert cached == result

    def test_cache_decorator(self):
        """Test caching decorator"""
        call_count = 0

        @cache_service.cached("test_category", ttl_seconds=60)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y

        # First call - should execute
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1

        # Second call with same args - should use cache
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Not incremented

        # Different args - should execute
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2


class TestRateLimiter:
    """Tests for rate limiting"""

    def test_basic_rate_limiting(self):
        """Test basic rate limiting functionality"""
        limiter = RateLimiter(
            requests_per_minute=5,
            requests_per_hour=20,
            requests_per_day=100
        )

        client_id = "test_client"

        # Should allow first 5 requests within a minute
        for i in range(5):
            allowed, info = limiter.check_rate_limit(client_id)
            assert allowed is True

        # 6th request should be blocked
        allowed, info = limiter.check_rate_limit(client_id)
        assert allowed is False
        assert info["retry_after"] > 0

    def test_rate_limit_reset(self):
        """Test that rate limits reset after time window"""
        limiter = RateLimiter(
            requests_per_minute=2,
            requests_per_hour=100,
            requests_per_day=1000
        )

        client_id = "test_client_2"

        # Use up minute quota
        limiter.check_rate_limit(client_id)
        limiter.check_rate_limit(client_id)

        # Should be blocked
        allowed, info = limiter.check_rate_limit(client_id)
        assert allowed is False

        # Wait for minute window to pass
        time.sleep(61)

        # Should be allowed again
        allowed, info = limiter.check_rate_limit(client_id)
        assert allowed is True

    def test_endpoint_specific_limits(self):
        """Test endpoint-specific rate limits"""
        limiter = RateLimiter(
            requests_per_minute=10,
            requests_per_hour=100,
            requests_per_day=1000,
            endpoint_limits={
                "/api/expensive": {"minute": 2, "hour": 10, "day": 50}
            }
        )

        client_id = "test_client_3"

        # Expensive endpoint should have stricter limits
        allowed, info = limiter.check_rate_limit(client_id, "/api/expensive")
        assert allowed is True

        allowed, info = limiter.check_rate_limit(client_id, "/api/expensive")
        assert allowed is True

        # 3rd request should be blocked
        allowed, info = limiter.check_rate_limit(client_id, "/api/expensive")
        assert allowed is False


class TestInputValidation:
    """Tests for input validation"""

    def test_sql_injection_detection(self):
        """Test SQL injection pattern detection"""
        validator = InputValidator()

        # Should detect SQL injection
        assert validator.check_sql_injection("SELECT * FROM users WHERE 1=1") is True
        assert validator.check_sql_injection("admin' OR '1'='1") is True
        assert validator.check_sql_injection("DROP TABLE users;") is True

        # Normal strings should pass
        assert validator.check_sql_injection("Find me a house") is False
        assert validator.check_sql_injection("3-bedroom property") is False

    def test_xss_detection(self):
        """Test XSS pattern detection"""
        validator = InputValidator()

        # Should detect XSS
        assert validator.check_xss("<script>alert('xss')</script>") is True
        assert validator.check_xss("<img src=x onerror=alert(1)>") is True
        assert validator.check_xss("javascript:alert(1)") is True

        # Normal HTML should be caught too (for safety)
        assert validator.check_xss("<iframe src='evil.com'>") is True

        # Normal strings should pass
        assert validator.check_xss("Modern 3-bedroom house") is False

    def test_command_injection_detection(self):
        """Test command injection pattern detection"""
        validator = InputValidator()

        # Should detect command injection
        assert validator.check_command_injection("; rm -rf /") is True
        assert validator.check_command_injection("| cat /etc/passwd") is True
        assert validator.check_command_injection("$(whoami)") is True

        # Normal strings should pass
        assert validator.check_command_injection("House with pool") is False

    def test_string_sanitization(self):
        """Test string sanitization"""
        validator = InputValidator()

        # Remove null bytes
        result = validator.sanitize_string("test\x00string")
        assert "\x00" not in result

        # Normalize whitespace
        result = validator.sanitize_string("test   string   with   spaces")
        assert result == "test string with spaces"

        # Trim to max length
        long_string = "a" * 100000
        validator.max_string_length = 1000
        result = validator.sanitize_string(long_string)
        assert len(result) == 1000

    def test_validate_dict(self):
        """Test dictionary validation"""
        validator = InputValidator()

        # Valid dict should pass
        data = {
            "query": "Find me a house",
            "limit": 10,
            "filters": {
                "bedrooms": 3,
                "price_max": 500000
            }
        }

        validated = validator.validate_dict(data)
        assert validated["query"] == "Find me a house"
        assert validated["limit"] == 10

    def test_validate_dict_with_attack(self):
        """Test dictionary validation with attack patterns"""
        validator = InputValidator()

        # Dict with SQL injection should be rejected
        data = {
            "query": "SELECT * FROM users WHERE 1=1",
            "limit": 10
        }

        with pytest.raises(Exception):  # Should raise HTTPException
            validator.validate_dict(data)


class TestAPIKeyManager:
    """Tests for API key management"""

    def test_create_and_validate_api_key(self):
        """Test creating and validating API keys"""
        manager = APIKeyManager()

        # Create API key
        api_key = manager.create_api_key(
            user_id="test_user",
            permissions=["workflow.execute", "query.search"],
            rate_limit_multiplier=2.0
        )

        assert api_key is not None
        assert api_key.startswith("rac_")

        # Validate API key
        key_info = manager.validate_api_key(api_key)

        assert key_info is not None
        assert key_info["user_id"] == "test_user"
        assert "workflow.execute" in key_info["permissions"]
        assert key_info["rate_limit_multiplier"] == 2.0

    def test_invalid_api_key(self):
        """Test validation of invalid API key"""
        manager = APIKeyManager()

        # Invalid key should return None
        key_info = manager.validate_api_key("invalid_key_12345")

        assert key_info is None

    def test_permission_checking(self):
        """Test permission checking"""
        manager = APIKeyManager()

        api_key = manager.create_api_key(
            user_id="test_user",
            permissions=["workflow.*", "query.search"],
            rate_limit_multiplier=1.0
        )

        key_info = manager.validate_api_key(api_key)

        # Exact match
        assert manager.check_permission(key_info, "query.search") is True

        # Wildcard match
        assert manager.check_permission(key_info, "workflow.execute") is True
        assert manager.check_permission(key_info, "workflow.list") is True

        # No match
        assert manager.check_permission(key_info, "admin.delete") is False

    def test_wildcard_permission(self):
        """Test wildcard permission"""
        manager = APIKeyManager()

        api_key = manager.create_api_key(
            user_id="admin_user",
            permissions=["*"],  # All permissions
            rate_limit_multiplier=10.0
        )

        key_info = manager.validate_api_key(api_key)

        # Should have access to everything
        assert manager.check_permission(key_info, "workflow.execute") is True
        assert manager.check_permission(key_info, "admin.delete") is True
        assert manager.check_permission(key_info, "any.permission") is True

    def test_revoke_api_key(self):
        """Test revoking API key"""
        manager = APIKeyManager()

        api_key = manager.create_api_key(
            user_id="test_user",
            permissions=["query.search"],
            rate_limit_multiplier=1.0
        )

        # Should be valid
        assert manager.validate_api_key(api_key) is not None

        # Revoke
        manager.revoke_api_key(api_key)

        # Should now be invalid
        assert manager.validate_api_key(api_key) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
