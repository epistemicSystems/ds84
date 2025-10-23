"""User analytics and telemetry service"""
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import json


@dataclass
class AnalyticsEvent:
    """Analytics event data"""
    event_id: str
    event_type: str
    user_id: str
    session_id: str
    timestamp: datetime
    properties: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


class AnalyticsService:
    """User analytics and telemetry tracking"""

    def __init__(self):
        """Initialize analytics service"""
        self.events: List[AnalyticsEvent] = []
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        self.feature_usage: Dict[str, int] = defaultdict(int)
        self.error_tracking: List[Dict[str, Any]] = []

    def track_event(
        self,
        event_type: str,
        user_id: str,
        session_id: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Track an analytics event

        Args:
            event_type: Type of event (e.g., "property_search", "workflow_executed")
            user_id: User identifier
            session_id: Optional session identifier
            properties: Event-specific properties
            context: Additional context (user agent, IP, etc.)

        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        session_id = session_id or str(uuid.uuid4())

        event = AnalyticsEvent(
            event_id=event_id,
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.now(),
            properties=properties or {},
            context=context or {}
        )

        self.events.append(event)

        # Track feature usage
        self.feature_usage[event_type] += 1

        # Update session tracking
        if session_id not in self.user_sessions:
            self.user_sessions[session_id] = {
                "session_id": session_id,
                "user_id": user_id,
                "start_time": datetime.now(),
                "last_activity": datetime.now(),
                "event_count": 0,
                "events": []
            }

        session = self.user_sessions[session_id]
        session["last_activity"] = datetime.now()
        session["event_count"] += 1
        session["events"].append(event_type)

        return event_id

    def track_page_view(
        self,
        user_id: str,
        page: str,
        session_id: Optional[str] = None,
        referrer: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> str:
        """Track a page view

        Args:
            user_id: User identifier
            page: Page path
            session_id: Session identifier
            referrer: Referrer URL
            user_agent: User agent string

        Returns:
            Event ID
        """
        return self.track_event(
            event_type="page_view",
            user_id=user_id,
            session_id=session_id,
            properties={"page": page},
            context={
                "referrer": referrer,
                "user_agent": user_agent
            }
        )

    def track_property_search(
        self,
        user_id: str,
        query: str,
        result_count: int,
        session_id: Optional[str] = None,
        duration_ms: Optional[float] = None
    ) -> str:
        """Track a property search

        Args:
            user_id: User identifier
            query: Search query
            result_count: Number of results returned
            session_id: Session identifier
            duration_ms: Search duration in milliseconds

        Returns:
            Event ID
        """
        return self.track_event(
            event_type="property_search",
            user_id=user_id,
            session_id=session_id,
            properties={
                "query": query,
                "result_count": result_count,
                "duration_ms": duration_ms
            }
        )

    def track_workflow_execution(
        self,
        user_id: str,
        workflow_id: str,
        execution_id: str,
        status: str,
        duration_ms: float,
        session_id: Optional[str] = None
    ) -> str:
        """Track a workflow execution

        Args:
            user_id: User identifier
            workflow_id: Workflow identifier
            execution_id: Execution identifier
            status: Execution status
            duration_ms: Duration in milliseconds
            session_id: Session identifier

        Returns:
            Event ID
        """
        return self.track_event(
            event_type="workflow_execution",
            user_id=user_id,
            session_id=session_id,
            properties={
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "status": status,
                "duration_ms": duration_ms
            }
        )

    def track_ab_test_assignment(
        self,
        user_id: str,
        test_id: str,
        variant_id: str,
        session_id: Optional[str] = None
    ) -> str:
        """Track A/B test variant assignment

        Args:
            user_id: User identifier
            test_id: Test identifier
            variant_id: Variant identifier
            session_id: Session identifier

        Returns:
            Event ID
        """
        return self.track_event(
            event_type="ab_test_assignment",
            user_id=user_id,
            session_id=session_id,
            properties={
                "test_id": test_id,
                "variant_id": variant_id
            }
        )

    def track_error(
        self,
        user_id: str,
        error_type: str,
        error_message: str,
        stack_trace: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> str:
        """Track an error

        Args:
            user_id: User identifier
            error_type: Type of error
            error_message: Error message
            stack_trace: Stack trace
            context: Error context
            session_id: Session identifier

        Returns:
            Event ID
        """
        error_data = {
            "user_id": user_id,
            "session_id": session_id,
            "error_type": error_type,
            "error_message": error_message,
            "stack_trace": stack_trace,
            "context": context or {},
            "timestamp": datetime.now()
        }

        self.error_tracking.append(error_data)

        return self.track_event(
            event_type="error",
            user_id=user_id,
            session_id=session_id,
            properties={
                "error_type": error_type,
                "error_message": error_message
            },
            context=context
        )

    def get_user_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get analytics for a specific user

        Args:
            user_id: User identifier
            days: Number of days to analyze

        Returns:
            User analytics summary
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        # Filter events for this user
        user_events = [
            event for event in self.events
            if event.user_id == user_id and event.timestamp >= cutoff_date
        ]

        if not user_events:
            return {
                "user_id": user_id,
                "total_events": 0,
                "sessions": 0,
                "message": "No events found for this user"
            }

        # Count events by type
        event_counts = Counter(event.event_type for event in user_events)

        # Get user sessions
        user_session_ids = set(event.session_id for event in user_events)
        sessions = [
            self.user_sessions[sid]
            for sid in user_session_ids
            if sid in self.user_sessions
        ]

        # Calculate engagement metrics
        first_event = min(user_events, key=lambda e: e.timestamp)
        last_event = max(user_events, key=lambda e: e.timestamp)

        return {
            "user_id": user_id,
            "total_events": len(user_events),
            "event_types": dict(event_counts),
            "sessions": len(sessions),
            "first_seen": first_event.timestamp.isoformat(),
            "last_seen": last_event.timestamp.isoformat(),
            "days_active": days,
            "avg_events_per_session": len(user_events) / len(sessions) if sessions else 0
        }

    def get_feature_usage(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get feature usage statistics

        Args:
            limit: Number of features to return

        Returns:
            List of feature usage stats
        """
        sorted_features = sorted(
            self.feature_usage.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            {"feature": feature, "usage_count": count}
            for feature, count in sorted_features[:limit]
        ]

    def get_active_users(self, minutes: int = 60) -> Dict[str, Any]:
        """Get active users in time window

        Args:
            minutes: Time window in minutes

        Returns:
            Active user statistics
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)

        # Count unique users with recent activity
        active_users = set(
            event.user_id for event in self.events
            if event.timestamp >= cutoff_time
        )

        # Count active sessions
        active_sessions = set(
            event.session_id for event in self.events
            if event.timestamp >= cutoff_time
        )

        return {
            "time_window_minutes": minutes,
            "active_users": len(active_users),
            "active_sessions": len(active_sessions),
            "timestamp": datetime.now().isoformat()
        }

    def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get analytics for a specific session

        Args:
            session_id: Session identifier

        Returns:
            Session analytics
        """
        if session_id not in self.user_sessions:
            return {
                "session_id": session_id,
                "found": False,
                "message": "Session not found"
            }

        session = self.user_sessions[session_id]
        duration = (session["last_activity"] - session["start_time"]).total_seconds()

        # Get events for this session
        session_events = [
            event for event in self.events
            if event.session_id == session_id
        ]

        event_counts = Counter(event.event_type for event in session_events)

        return {
            "session_id": session_id,
            "found": True,
            "user_id": session["user_id"],
            "start_time": session["start_time"].isoformat(),
            "last_activity": session["last_activity"].isoformat(),
            "duration_seconds": round(duration, 2),
            "event_count": session["event_count"],
            "event_types": dict(event_counts),
            "events": session["events"]
        }

    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for time period

        Args:
            hours: Time window in hours

        Returns:
            Error summary statistics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_errors = [
            error for error in self.error_tracking
            if error["timestamp"] >= cutoff_time
        ]

        if not recent_errors:
            return {
                "time_window_hours": hours,
                "total_errors": 0,
                "error_types": {}
            }

        error_type_counts = Counter(error["error_type"] for error in recent_errors)

        # Get unique users affected
        affected_users = set(error["user_id"] for error in recent_errors)

        return {
            "time_window_hours": hours,
            "total_errors": len(recent_errors),
            "unique_users_affected": len(affected_users),
            "error_types": dict(error_type_counts),
            "recent_errors": [
                {
                    "error_type": error["error_type"],
                    "error_message": error["error_message"],
                    "timestamp": error["timestamp"].isoformat(),
                    "user_id": error["user_id"]
                }
                for error in recent_errors[-10:]  # Last 10 errors
            ]
        }

    def get_conversion_funnel(
        self,
        funnel_steps: List[str],
        days: int = 7
    ) -> Dict[str, Any]:
        """Analyze conversion funnel

        Args:
            funnel_steps: List of event types representing funnel steps
            days: Number of days to analyze

        Returns:
            Funnel analysis
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        # Get users who completed each step
        step_users = {}
        for step in funnel_steps:
            step_users[step] = set(
                event.user_id for event in self.events
                if event.event_type == step and event.timestamp >= cutoff_date
            )

        # Calculate conversion rates
        funnel_data = []
        previous_count = None

        for i, step in enumerate(funnel_steps):
            count = len(step_users[step])

            if i == 0:
                conversion_rate = 1.0
                dropoff_rate = 0.0
            else:
                conversion_rate = count / previous_count if previous_count > 0 else 0.0
                dropoff_rate = 1.0 - conversion_rate

            funnel_data.append({
                "step": i + 1,
                "event_type": step,
                "users": count,
                "conversion_rate": round(conversion_rate, 4),
                "dropoff_rate": round(dropoff_rate, 4)
            })

            previous_count = count

        return {
            "funnel_steps": funnel_steps,
            "time_window_days": days,
            "analysis": funnel_data,
            "overall_conversion": round(
                len(step_users[funnel_steps[-1]]) / len(step_users[funnel_steps[0]])
                if len(step_users[funnel_steps[0]]) > 0 else 0.0,
                4
            )
        }

    def export_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Export events as JSON

        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            event_type: Optional event type filter

        Returns:
            List of event dictionaries
        """
        filtered_events = self.events

        if start_date:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_date]

        if end_date:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_date]

        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]

        return [
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "user_id": event.user_id,
                "session_id": event.session_id,
                "timestamp": event.timestamp.isoformat(),
                "properties": event.properties,
                "context": event.context
            }
            for event in filtered_events
        ]


# Global analytics service instance
analytics_service = AnalyticsService()
