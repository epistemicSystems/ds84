"""Interaction logging system for capturing user behavior and learning patterns"""
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class InteractionType(str, Enum):
    """Types of user interactions"""
    QUERY = "query"
    FEEDBACK = "feedback"
    CLICK = "click"
    REFINEMENT = "refinement"
    ERROR = "error"
    SESSION_START = "session_start"
    SESSION_END = "session_end"


class FeedbackType(str, Enum):
    """Types of user feedback"""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    PROPERTY_VIEWED = "property_viewed"
    PROPERTY_SAVED = "property_saved"
    PROPERTY_SHARED = "property_shared"
    QUERY_REFINED = "query_refined"
    RESULT_CLICKED = "result_clicked"


@dataclass
class Interaction:
    """Represents a single user interaction"""
    interaction_id: str
    session_id: str
    user_id: str  # Anonymized
    interaction_type: InteractionType
    timestamp: datetime
    data: Dict[str, Any]
    context: Dict[str, Any]
    privacy_level: str  # public, private, anonymous


@dataclass
class QueryInteraction:
    """Specific interaction for queries"""
    query_text: str  # Optionally anonymized
    intent_extracted: Dict[str, Any]
    results_count: int
    response_time_ms: float
    search_mode: str
    user_feedback: Optional[str] = None


@dataclass
class FeedbackInteraction:
    """Specific interaction for feedback"""
    feedback_type: FeedbackType
    related_query_id: Optional[str]
    related_item_id: Optional[str]  # property_id, agent_id, etc.
    rating: Optional[float]  # 1-5 scale
    comment: Optional[str]


@dataclass
class EngagementMetrics:
    """Metrics for user engagement"""
    session_duration_seconds: float
    queries_per_session: int
    results_clicked: int
    refinements_made: int
    properties_saved: int
    feedback_provided: int
    error_rate: float


class InteractionLogger:
    """Privacy-preserving interaction logging system"""

    def __init__(self, log_dir: str = "logs/interactions"):
        """Initialize interaction logger

        Args:
            log_dir: Directory to store interaction logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Privacy settings
        self.anonymize_queries = True  # Hash sensitive query data
        self.retain_days = 90  # How long to keep logs
        self.pii_fields = ["email", "phone", "address"]  # Fields to anonymize

        # In-memory cache for session tracking
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

    def log_interaction(
        self,
        session_id: str,
        user_id: str,
        interaction_type: InteractionType,
        data: Dict[str, Any],
        context: Dict[str, Any] = None,
        privacy_level: str = "private"
    ) -> str:
        """Log a user interaction

        Args:
            session_id: Session identifier
            user_id: User identifier (will be anonymized)
            interaction_type: Type of interaction
            data: Interaction data
            context: Additional context
            privacy_level: Privacy level for this interaction

        Returns:
            Interaction ID
        """
        # Generate interaction ID
        interaction_id = self._generate_id(session_id, user_id, interaction_type.value)

        # Anonymize user ID
        anonymized_user_id = self._anonymize_id(user_id)

        # Sanitize data for privacy
        sanitized_data = self._sanitize_data(data, privacy_level)
        sanitized_context = self._sanitize_data(context or {}, privacy_level)

        # Create interaction
        interaction = Interaction(
            interaction_id=interaction_id,
            session_id=session_id,
            user_id=anonymized_user_id,
            interaction_type=interaction_type,
            timestamp=datetime.utcnow(),
            data=sanitized_data,
            context=sanitized_context,
            privacy_level=privacy_level
        )

        # Write to log
        self._write_log(interaction)

        # Update session tracking
        self._update_session(session_id, interaction)

        return interaction_id

    def log_query(
        self,
        session_id: str,
        user_id: str,
        query_text: str,
        intent_extracted: Dict[str, Any],
        results_count: int,
        response_time_ms: float,
        search_mode: str
    ) -> str:
        """Log a property search query

        Args:
            session_id: Session identifier
            user_id: User identifier
            query_text: Query text
            intent_extracted: Extracted intent
            results_count: Number of results
            response_time_ms: Response time
            search_mode: Search mode used

        Returns:
            Interaction ID
        """
        query_data = QueryInteraction(
            query_text=query_text if not self.anonymize_queries else self._hash_text(query_text),
            intent_extracted=intent_extracted,
            results_count=results_count,
            response_time_ms=response_time_ms,
            search_mode=search_mode
        )

        return self.log_interaction(
            session_id=session_id,
            user_id=user_id,
            interaction_type=InteractionType.QUERY,
            data=asdict(query_data),
            context={"search_mode": search_mode},
            privacy_level="private"
        )

    def log_feedback(
        self,
        session_id: str,
        user_id: str,
        feedback_type: FeedbackType,
        related_query_id: Optional[str] = None,
        related_item_id: Optional[str] = None,
        rating: Optional[float] = None,
        comment: Optional[str] = None
    ) -> str:
        """Log user feedback

        Args:
            session_id: Session identifier
            user_id: User identifier
            feedback_type: Type of feedback
            related_query_id: Related query ID
            related_item_id: Related item ID
            rating: Rating (1-5)
            comment: Feedback comment

        Returns:
            Interaction ID
        """
        feedback_data = FeedbackInteraction(
            feedback_type=feedback_type,
            related_query_id=related_query_id,
            related_item_id=related_item_id,
            rating=rating,
            comment=comment
        )

        return self.log_interaction(
            session_id=session_id,
            user_id=user_id,
            interaction_type=InteractionType.FEEDBACK,
            data=asdict(feedback_data),
            context={"feedback_type": feedback_type.value},
            privacy_level="private"
        )

    def log_error(
        self,
        session_id: str,
        user_id: str,
        error_type: str,
        error_message: str,
        context: Dict[str, Any] = None
    ) -> str:
        """Log an error interaction

        Args:
            session_id: Session identifier
            user_id: User identifier
            error_type: Type of error
            error_message: Error message
            context: Error context

        Returns:
            Interaction ID
        """
        error_data = {
            "error_type": error_type,
            "error_message": error_message,
            "stack_trace": context.get("stack_trace", "") if context else ""
        }

        return self.log_interaction(
            session_id=session_id,
            user_id=user_id,
            interaction_type=InteractionType.ERROR,
            data=error_data,
            context=context or {},
            privacy_level="private"
        )

    def start_session(
        self,
        session_id: str,
        user_id: str,
        metadata: Dict[str, Any] = None
    ):
        """Start a user session

        Args:
            session_id: Session identifier
            user_id: User identifier
            metadata: Session metadata
        """
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "started_at": datetime.utcnow(),
            "interactions": [],
            "metadata": metadata or {}
        }

        self.log_interaction(
            session_id=session_id,
            user_id=user_id,
            interaction_type=InteractionType.SESSION_START,
            data=metadata or {},
            privacy_level="anonymous"
        )

    def end_session(
        self,
        session_id: str,
        user_id: str
    ) -> EngagementMetrics:
        """End a user session and calculate metrics

        Args:
            session_id: Session identifier
            user_id: User identifier

        Returns:
            Engagement metrics for the session
        """
        if session_id not in self.active_sessions:
            # Session not tracked, return empty metrics
            return EngagementMetrics(
                session_duration_seconds=0,
                queries_per_session=0,
                results_clicked=0,
                refinements_made=0,
                properties_saved=0,
                feedback_provided=0,
                error_rate=0.0
            )

        session = self.active_sessions[session_id]
        interactions = session.get("interactions", [])

        # Calculate metrics
        started_at = session.get("started_at", datetime.utcnow())
        duration = (datetime.utcnow() - started_at).total_seconds()

        metrics = EngagementMetrics(
            session_duration_seconds=duration,
            queries_per_session=sum(1 for i in interactions if i.get("type") == "query"),
            results_clicked=sum(1 for i in interactions if i.get("type") == "click"),
            refinements_made=sum(1 for i in interactions if i.get("type") == "refinement"),
            properties_saved=sum(1 for i in interactions if i.get("type") == "feedback" and i.get("data", {}).get("feedback_type") == "property_saved"),
            feedback_provided=sum(1 for i in interactions if i.get("type") == "feedback"),
            error_rate=sum(1 for i in interactions if i.get("type") == "error") / len(interactions) if interactions else 0.0
        )

        # Log session end with metrics
        self.log_interaction(
            session_id=session_id,
            user_id=user_id,
            interaction_type=InteractionType.SESSION_END,
            data=asdict(metrics),
            privacy_level="anonymous"
        )

        # Clean up session
        del self.active_sessions[session_id]

        return metrics

    def get_session_interactions(
        self,
        session_id: str
    ) -> List[Dict[str, Any]]:
        """Get all interactions for a session

        Args:
            session_id: Session identifier

        Returns:
            List of interactions
        """
        if session_id in self.active_sessions:
            return self.active_sessions[session_id].get("interactions", [])

        # Load from logs (would query database in production)
        return []

    def get_user_history(
        self,
        user_id: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get interaction history for a user

        Args:
            user_id: User identifier
            days: Number of days to look back

        Returns:
            List of interactions
        """
        anonymized_user_id = self._anonymize_id(user_id)
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Load interactions from logs
        interactions = []
        for log_file in self.log_dir.glob("interactions_*.jsonl"):
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        if (data.get("user_id") == anonymized_user_id and
                            datetime.fromisoformat(data.get("timestamp", "")) >= cutoff_date):
                            interactions.append(data)
            except Exception:
                continue

        return interactions

    def cleanup_old_logs(self, days: Optional[int] = None):
        """Delete logs older than retention period

        Args:
            days: Number of days to retain (uses default if None)
        """
        retention_days = days or self.retain_days
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)

        for log_file in self.log_dir.glob("interactions_*.jsonl"):
            try:
                # Check file modification time
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_time < cutoff_date:
                    log_file.unlink()
                    print(f"Deleted old log file: {log_file}")
            except Exception as e:
                print(f"Error cleaning up {log_file}: {e}")

    def _generate_id(self, session_id: str, user_id: str, interaction_type: str) -> str:
        """Generate unique interaction ID"""
        timestamp = datetime.utcnow().isoformat()
        data = f"{session_id}:{user_id}:{interaction_type}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _anonymize_id(self, user_id: str) -> str:
        """Anonymize user ID using one-way hash"""
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]

    def _hash_text(self, text: str) -> str:
        """Hash text for privacy"""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _sanitize_data(
        self,
        data: Dict[str, Any],
        privacy_level: str
    ) -> Dict[str, Any]:
        """Sanitize data based on privacy level

        Args:
            data: Data to sanitize
            privacy_level: Privacy level

        Returns:
            Sanitized data
        """
        if privacy_level == "public":
            # Public data - minimal sanitization
            return data

        sanitized = {}
        for key, value in data.items():
            # Remove PII fields
            if key.lower() in self.pii_fields:
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_data(value, privacy_level)
            else:
                sanitized[key] = value

        return sanitized

    def _write_log(self, interaction: Interaction):
        """Write interaction to log file

        Args:
            interaction: Interaction to log
        """
        # Daily log file
        today = datetime.utcnow().strftime("%Y-%m-%d")
        log_file = self.log_dir / f"interactions_{today}.jsonl"

        # Convert to dict
        log_entry = {
            "interaction_id": interaction.interaction_id,
            "session_id": interaction.session_id,
            "user_id": interaction.user_id,
            "interaction_type": interaction.interaction_type.value,
            "timestamp": interaction.timestamp.isoformat(),
            "data": interaction.data,
            "context": interaction.context,
            "privacy_level": interaction.privacy_level
        }

        # Append to log file (JSON Lines format)
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def _update_session(self, session_id: str, interaction: Interaction):
        """Update session tracking

        Args:
            session_id: Session identifier
            interaction: Interaction to add
        """
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["interactions"].append({
                "type": interaction.interaction_type.value,
                "timestamp": interaction.timestamp.isoformat(),
                "data": interaction.data
            })


# Global instance
interaction_logger = InteractionLogger()
