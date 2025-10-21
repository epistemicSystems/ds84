"""Context management for cognitive workflows"""
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import deque
from pydantic import BaseModel


class Message(BaseModel):
    """Conversation message"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = None
    metadata: Dict[str, Any] = {}

    def __init__(self, **data):
        if data.get('timestamp') is None:
            data['timestamp'] = datetime.now()
        super().__init__(**data)


class UserPreference(BaseModel):
    """User preference entry"""
    key: str
    value: Any
    confidence: float = 1.0  # 0.0 to 1.0
    source: str = "explicit"  # "explicit" or "inferred"
    updated_at: datetime = None

    def __init__(self, **data):
        if data.get('updated_at') is None:
            data['updated_at'] = datetime.now()
        super().__init__(**data)


class SessionContext(BaseModel):
    """Session-level context"""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    messages: List[Message] = []
    metadata: Dict[str, Any] = {}


class UserContext(BaseModel):
    """User-level context"""
    user_id: str
    preferences: Dict[str, UserPreference] = {}
    long_term_context: Dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime


class ContextManager:
    """Manages conversation history, user preferences, and context"""

    def __init__(
        self,
        max_session_messages: int = 50,
        session_timeout_hours: int = 24,
        max_context_tokens: int = 4000
    ):
        self.max_session_messages = max_session_messages
        self.session_timeout = timedelta(hours=session_timeout_hours)
        self.max_context_tokens = max_context_tokens

        # In-memory storage (in production, use database)
        self.sessions: Dict[str, SessionContext] = {}
        self.users: Dict[str, UserContext] = {}

    def create_session(self, session_id: str, user_id: str) -> SessionContext:
        """Create a new session

        Args:
            session_id: Session identifier
            user_id: User identifier

        Returns:
            New session context
        """
        now = datetime.now()

        session = SessionContext(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_activity=now
        )

        self.sessions[session_id] = session

        # Ensure user exists
        if user_id not in self.users:
            self.users[user_id] = UserContext(
                user_id=user_id,
                created_at=now,
                updated_at=now
            )

        return session

    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Get session by ID

        Args:
            session_id: Session identifier

        Returns:
            Session context or None
        """
        session = self.sessions.get(session_id)

        if session:
            # Check if session expired
            if datetime.now() - session.last_activity > self.session_timeout:
                # Session expired, remove it
                del self.sessions[session_id]
                return None

        return session

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None
    ):
        """Add message to session conversation history

        Args:
            session_id: Session identifier
            role: Message role ("user" or "assistant")
            content: Message content
            metadata: Optional message metadata
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )

        session.messages.append(message)
        session.last_activity = datetime.now()

        # Trim messages if exceeds max
        if len(session.messages) > self.max_session_messages:
            session.messages = session.messages[-self.max_session_messages:]

    def get_conversation_history(
        self,
        session_id: str,
        limit: int = None
    ) -> List[Message]:
        """Get conversation history for session

        Args:
            session_id: Session identifier
            limit: Optional limit on number of messages

        Returns:
            List of messages
        """
        session = self.get_session(session_id)
        if not session:
            return []

        messages = session.messages

        if limit:
            messages = messages[-limit:]

        return messages

    def get_context_window(
        self,
        session_id: str,
        max_tokens: int = None
    ) -> str:
        """Get conversation context as formatted string

        Args:
            session_id: Session identifier
            max_tokens: Maximum tokens (approximate)

        Returns:
            Formatted context string
        """
        messages = self.get_conversation_history(session_id)

        if not messages:
            return ""

        max_tokens = max_tokens or self.max_context_tokens

        # Build context string
        context_parts = []
        total_tokens = 0

        for message in reversed(messages):
            msg_text = f"{message.role.upper()}: {message.content}"
            msg_tokens = len(msg_text.split())  # Rough estimate

            if total_tokens + msg_tokens > max_tokens:
                break

            context_parts.insert(0, msg_text)
            total_tokens += msg_tokens

        return "\n\n".join(context_parts)

    def set_user_preference(
        self,
        user_id: str,
        key: str,
        value: Any,
        confidence: float = 1.0,
        source: str = "explicit"
    ):
        """Set user preference

        Args:
            user_id: User identifier
            key: Preference key
            value: Preference value
            confidence: Confidence level (0.0 to 1.0)
            source: "explicit" or "inferred"
        """
        if user_id not in self.users:
            self.users[user_id] = UserContext(
                user_id=user_id,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )

        preference = UserPreference(
            key=key,
            value=value,
            confidence=confidence,
            source=source
        )

        self.users[user_id].preferences[key] = preference
        self.users[user_id].updated_at = datetime.now()

    def get_user_preference(
        self,
        user_id: str,
        key: str,
        default: Any = None
    ) -> Any:
        """Get user preference

        Args:
            user_id: User identifier
            key: Preference key
            default: Default value if not found

        Returns:
            Preference value or default
        """
        user = self.users.get(user_id)
        if not user:
            return default

        preference = user.preferences.get(key)
        if not preference:
            return default

        return preference.value

    def get_all_preferences(
        self,
        user_id: str,
        min_confidence: float = 0.0
    ) -> Dict[str, Any]:
        """Get all user preferences

        Args:
            user_id: User identifier
            min_confidence: Minimum confidence threshold

        Returns:
            Dictionary of preferences
        """
        user = self.users.get(user_id)
        if not user:
            return {}

        return {
            key: pref.value
            for key, pref in user.preferences.items()
            if pref.confidence >= min_confidence
        }

    def infer_preferences_from_interaction(
        self,
        user_id: str,
        query: str,
        selected_properties: List[Dict[str, Any]] = None,
        feedback: str = None
    ):
        """Infer user preferences from interaction

        Args:
            user_id: User identifier
            query: User query
            selected_properties: Properties user showed interest in
            feedback: User feedback
        """
        # Extract preferences from query
        query_lower = query.lower()

        # Location preferences
        if "near" in query_lower or "close to" in query_lower:
            # Extract location features
            pass

        # Property type preferences
        property_types = []
        if "house" in query_lower:
            property_types.append("house")
        if "condo" in query_lower:
            property_types.append("condo")

        if property_types:
            current = self.get_user_preference(user_id, "preferred_property_types", [])
            if not isinstance(current, list):
                current = []
            updated = list(set(current + property_types))
            self.set_user_preference(
                user_id,
                "preferred_property_types",
                updated,
                confidence=0.7,
                source="inferred"
            )

        # Learn from selected properties
        if selected_properties:
            # Extract common features
            price_range = [p.get('price', 0) for p in selected_properties]
            if price_range:
                avg_price = sum(price_range) / len(price_range)
                self.set_user_preference(
                    user_id,
                    "inferred_budget",
                    int(avg_price),
                    confidence=0.6,
                    source="inferred"
                )

    def set_long_term_context(
        self,
        user_id: str,
        key: str,
        value: Any
    ):
        """Set long-term context for user

        Args:
            user_id: User identifier
            key: Context key
            value: Context value
        """
        if user_id not in self.users:
            self.users[user_id] = UserContext(
                user_id=user_id,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )

        self.users[user_id].long_term_context[key] = value
        self.users[user_id].updated_at = datetime.now()

    def get_long_term_context(
        self,
        user_id: str,
        key: str = None
    ) -> Any:
        """Get long-term context for user

        Args:
            user_id: User identifier
            key: Optional specific context key

        Returns:
            Context value or entire context dict
        """
        user = self.users.get(user_id)
        if not user:
            return {} if key is None else None

        if key is None:
            return user.long_term_context

        return user.long_term_context.get(key)

    def get_full_context(
        self,
        session_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Get complete context for workflow execution

        Args:
            session_id: Session identifier
            user_id: User identifier

        Returns:
            Complete context dictionary
        """
        return {
            "conversation_history": self.get_context_window(session_id),
            "recent_messages": [
                msg.dict() for msg in self.get_conversation_history(session_id, limit=5)
            ],
            "user_preferences": self.get_all_preferences(user_id),
            "long_term_context": self.get_long_term_context(user_id)
        }


# Global context manager instance
context_manager = ContextManager()
