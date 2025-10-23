"""Feature flags system for controlled feature rollouts"""
import hashlib
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class RolloutStrategy(Enum):
    """Feature rollout strategy"""
    ALL_USERS = "all_users"
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    USER_ATTRIBUTE = "user_attribute"
    GRADUAL = "gradual"


@dataclass
class FeatureFlag:
    """Feature flag configuration"""
    flag_id: str
    name: str
    description: str
    enabled: bool = False
    strategy: RolloutStrategy = RolloutStrategy.ALL_USERS
    percentage: float = 0.0  # For percentage rollout (0.0 to 100.0)
    user_whitelist: Set[str] = field(default_factory=set)
    user_blacklist: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"


class FeatureFlagService:
    """Service for managing feature flags"""

    def __init__(self):
        """Initialize feature flag service"""
        self.flags: Dict[str, FeatureFlag] = {}
        self._initialize_default_flags()

    def _initialize_default_flags(self):
        """Initialize default feature flags"""
        # A/B Testing
        self.create_flag(
            flag_id="ab_testing",
            name="A/B Testing Framework",
            description="Enable A/B testing capabilities",
            enabled=True
        )

        # Feedback Learning
        self.create_flag(
            flag_id="feedback_learning",
            name="Feedback Learning",
            description="Enable feedback learning and personalization",
            enabled=True
        )

        # Meta-cognitive Optimization
        self.create_flag(
            flag_id="meta_cognitive",
            name="Meta-cognitive Optimization",
            description="Enable recursive self-improvement and optimization",
            enabled=True
        )

        # Advanced Caching
        self.create_flag(
            flag_id="advanced_caching",
            name="Advanced Caching",
            description="Enable advanced caching strategies",
            enabled=True
        )

        # Performance Monitoring
        self.create_flag(
            flag_id="performance_monitoring",
            name="Performance Monitoring",
            description="Enable detailed performance monitoring",
            enabled=True
        )

        # User Analytics
        self.create_flag(
            flag_id="user_analytics",
            name="User Analytics",
            description="Enable user analytics and telemetry",
            enabled=True
        )

        # Experimental Features
        self.create_flag(
            flag_id="experimental_features",
            name="Experimental Features",
            description="Enable experimental and beta features",
            enabled=False,
            strategy=RolloutStrategy.PERCENTAGE,
            percentage=5.0  # 5% of users
        )

        # GPT-4 for All
        self.create_flag(
            flag_id="gpt4_all_workflows",
            name="GPT-4 for All Workflows",
            description="Use GPT-4 for all workflow executions",
            enabled=False,
            strategy=RolloutStrategy.PERCENTAGE,
            percentage=10.0  # 10% of users initially
        )

    def create_flag(
        self,
        flag_id: str,
        name: str,
        description: str,
        enabled: bool = False,
        strategy: RolloutStrategy = RolloutStrategy.ALL_USERS,
        percentage: float = 0.0,
        user_whitelist: Optional[Set[str]] = None,
        user_blacklist: Optional[Set[str]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        created_by: str = "system"
    ) -> FeatureFlag:
        """Create a new feature flag

        Args:
            flag_id: Unique flag identifier
            name: Human-readable name
            description: Flag description
            enabled: Whether flag is enabled
            strategy: Rollout strategy
            percentage: Percentage for gradual rollout
            user_whitelist: Users who always get the feature
            user_blacklist: Users who never get the feature
            attributes: Additional attributes
            created_by: Creator identifier

        Returns:
            Created feature flag
        """
        flag = FeatureFlag(
            flag_id=flag_id,
            name=name,
            description=description,
            enabled=enabled,
            strategy=strategy,
            percentage=percentage,
            user_whitelist=user_whitelist or set(),
            user_blacklist=user_blacklist or set(),
            attributes=attributes or {},
            created_by=created_by
        )

        self.flags[flag_id] = flag
        return flag

    def update_flag(
        self,
        flag_id: str,
        enabled: Optional[bool] = None,
        strategy: Optional[RolloutStrategy] = None,
        percentage: Optional[float] = None,
        user_whitelist: Optional[Set[str]] = None,
        user_blacklist: Optional[Set[str]] = None
    ) -> FeatureFlag:
        """Update an existing feature flag

        Args:
            flag_id: Flag identifier
            enabled: Whether flag is enabled
            strategy: Rollout strategy
            percentage: Percentage for gradual rollout
            user_whitelist: Users who always get the feature
            user_blacklist: Users who never get the feature

        Returns:
            Updated feature flag

        Raises:
            ValueError: If flag not found
        """
        if flag_id not in self.flags:
            raise ValueError(f"Flag {flag_id} not found")

        flag = self.flags[flag_id]

        if enabled is not None:
            flag.enabled = enabled

        if strategy is not None:
            flag.strategy = strategy

        if percentage is not None:
            flag.percentage = min(100.0, max(0.0, percentage))

        if user_whitelist is not None:
            flag.user_whitelist = user_whitelist

        if user_blacklist is not None:
            flag.user_blacklist = user_blacklist

        flag.updated_at = datetime.now()

        return flag

    def delete_flag(self, flag_id: str) -> bool:
        """Delete a feature flag

        Args:
            flag_id: Flag identifier

        Returns:
            True if deleted, False if not found
        """
        if flag_id in self.flags:
            del self.flags[flag_id]
            return True
        return False

    def is_enabled(
        self,
        flag_id: str,
        user_id: Optional[str] = None,
        user_attributes: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if feature is enabled for user

        Args:
            flag_id: Flag identifier
            user_id: User identifier
            user_attributes: User attributes for targeting

        Returns:
            True if feature is enabled for this user
        """
        if flag_id not in self.flags:
            # Unknown flags default to disabled
            return False

        flag = self.flags[flag_id]

        # If flag is globally disabled, return False
        if not flag.enabled:
            return False

        # If no user_id provided, check global flag only
        if user_id is None:
            return flag.strategy == RolloutStrategy.ALL_USERS

        # Check blacklist first
        if user_id in flag.user_blacklist:
            return False

        # Check whitelist
        if user_id in flag.user_whitelist:
            return True

        # Apply rollout strategy
        if flag.strategy == RolloutStrategy.ALL_USERS:
            return True

        elif flag.strategy == RolloutStrategy.PERCENTAGE:
            return self._percentage_rollout(user_id, flag_id, flag.percentage)

        elif flag.strategy == RolloutStrategy.USER_LIST:
            return user_id in flag.user_whitelist

        elif flag.strategy == RolloutStrategy.USER_ATTRIBUTE:
            return self._attribute_match(user_attributes, flag.attributes)

        elif flag.strategy == RolloutStrategy.GRADUAL:
            # Gradual rollout based on percentage
            return self._percentage_rollout(user_id, flag_id, flag.percentage)

        return False

    def _percentage_rollout(self, user_id: str, flag_id: str, percentage: float) -> bool:
        """Determine if user is in percentage rollout

        Uses consistent hashing to ensure same user always gets same result

        Args:
            user_id: User identifier
            flag_id: Flag identifier
            percentage: Target percentage (0.0 to 100.0)

        Returns:
            True if user is in rollout percentage
        """
        # Create consistent hash from user_id and flag_id
        hash_input = f"{flag_id}:{user_id}".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)

        # Convert to percentage (0-100)
        user_percentage = (hash_value % 10000) / 100.0

        return user_percentage < percentage

    def _attribute_match(
        self,
        user_attributes: Optional[Dict[str, Any]],
        required_attributes: Dict[str, Any]
    ) -> bool:
        """Check if user attributes match required attributes

        Args:
            user_attributes: User's attributes
            required_attributes: Required attribute values

        Returns:
            True if all required attributes match
        """
        if not user_attributes or not required_attributes:
            return False

        for key, value in required_attributes.items():
            if key not in user_attributes:
                return False

            if isinstance(value, list):
                # Check if user value is in list
                if user_attributes[key] not in value:
                    return False
            else:
                # Exact match
                if user_attributes[key] != value:
                    return False

        return True

    def get_flag(self, flag_id: str) -> Optional[FeatureFlag]:
        """Get feature flag by ID

        Args:
            flag_id: Flag identifier

        Returns:
            Feature flag or None if not found
        """
        return self.flags.get(flag_id)

    def list_flags(self, enabled_only: bool = False) -> List[FeatureFlag]:
        """List all feature flags

        Args:
            enabled_only: Only return enabled flags

        Returns:
            List of feature flags
        """
        flags = list(self.flags.values())

        if enabled_only:
            flags = [f for f in flags if f.enabled]

        return sorted(flags, key=lambda f: f.flag_id)

    def get_user_flags(
        self,
        user_id: str,
        user_attributes: Optional[Dict[str, Any]] = None
    ) -> Dict[str, bool]:
        """Get all flags enabled for a specific user

        Args:
            user_id: User identifier
            user_attributes: User attributes

        Returns:
            Dictionary mapping flag IDs to enabled status
        """
        return {
            flag_id: self.is_enabled(flag_id, user_id, user_attributes)
            for flag_id in self.flags.keys()
        }

    def get_flag_stats(self) -> Dict[str, Any]:
        """Get statistics about feature flags

        Returns:
            Flag statistics
        """
        total_flags = len(self.flags)
        enabled_flags = sum(1 for f in self.flags.values() if f.enabled)

        strategy_counts = {}
        for flag in self.flags.values():
            strategy = flag.strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        return {
            "total_flags": total_flags,
            "enabled_flags": enabled_flags,
            "disabled_flags": total_flags - enabled_flags,
            "strategies": strategy_counts,
            "flags": [
                {
                    "flag_id": flag.flag_id,
                    "name": flag.name,
                    "enabled": flag.enabled,
                    "strategy": flag.strategy.value,
                    "percentage": flag.percentage if flag.strategy == RolloutStrategy.PERCENTAGE else None
                }
                for flag in self.flags.values()
            ]
        }

    def gradual_rollout(
        self,
        flag_id: str,
        target_percentage: float,
        increment: float = 5.0
    ) -> Dict[str, Any]:
        """Gradually increase rollout percentage

        Args:
            flag_id: Flag identifier
            target_percentage: Target percentage
            increment: Percentage increment per step

        Returns:
            Rollout plan
        """
        if flag_id not in self.flags:
            raise ValueError(f"Flag {flag_id} not found")

        flag = self.flags[flag_id]
        current = flag.percentage

        if current >= target_percentage:
            return {
                "flag_id": flag_id,
                "current_percentage": current,
                "target_percentage": target_percentage,
                "message": "Already at or above target percentage"
            }

        # Calculate rollout steps
        steps = []
        percentage = current

        while percentage < target_percentage:
            percentage = min(percentage + increment, target_percentage)
            steps.append(percentage)

        return {
            "flag_id": flag_id,
            "current_percentage": current,
            "target_percentage": target_percentage,
            "increment": increment,
            "steps": steps,
            "total_steps": len(steps),
            "recommendation": f"Gradually increase from {current}% to {target_percentage}% in {len(steps)} steps"
        }


# Global feature flag service instance
feature_flags = FeatureFlagService()
