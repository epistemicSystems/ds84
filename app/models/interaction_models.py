"""Database models for interaction tracking and feedback"""
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum


class InteractionTypeEnum(str, Enum):
    """Types of user interactions"""
    QUERY = "query"
    FEEDBACK = "feedback"
    CLICK = "click"
    REFINEMENT = "refinement"
    ERROR = "error"
    SESSION_START = "session_start"
    SESSION_END = "session_end"


class FeedbackTypeEnum(str, Enum):
    """Types of user feedback"""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    PROPERTY_VIEWED = "property_viewed"
    PROPERTY_SAVED = "property_saved"
    PROPERTY_SHARED = "property_shared"
    QUERY_REFINED = "query_refined"
    RESULT_CLICKED = "result_clicked"


class InteractionRecord(BaseModel):
    """Database model for interaction records"""
    interaction_id: str
    session_id: str
    user_id: str  # Anonymized
    interaction_type: InteractionTypeEnum
    timestamp: datetime
    data: Dict[str, Any]
    context: Dict[str, Any] = {}
    privacy_level: str = "private"

    class Config:
        use_enum_values = True


class QueryRecord(BaseModel):
    """Record of a search query"""
    query_id: str
    session_id: str
    user_id: str
    query_text: str  # May be hashed
    intent_extracted: Dict[str, Any]
    results_count: int
    response_time_ms: float
    search_mode: str
    timestamp: datetime
    feedback_received: bool = False


class FeedbackRecord(BaseModel):
    """Record of user feedback"""
    feedback_id: str
    session_id: str
    user_id: str
    feedback_type: FeedbackTypeEnum
    related_query_id: Optional[str] = None
    related_item_id: Optional[str] = None
    rating: Optional[float] = None
    comment: Optional[str] = None
    timestamp: datetime

    class Config:
        use_enum_values = True


class SessionRecord(BaseModel):
    """Record of a user session"""
    session_id: str
    user_id: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    interactions_count: int = 0
    queries_count: int = 0
    feedback_count: int = 0
    errors_count: int = 0
    metadata: Dict[str, Any] = {}


class EngagementMetricsRecord(BaseModel):
    """Engagement metrics for a session"""
    session_id: str
    user_id: str
    session_duration_seconds: float
    queries_per_session: int
    results_clicked: int
    refinements_made: int
    properties_saved: int
    feedback_provided: int
    error_rate: float
    timestamp: datetime


class UserPreferenceRecord(BaseModel):
    """Learned user preferences"""
    user_id: str
    preference_type: str  # property_features, location, price_range, etc.
    preference_value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    learned_from: str  # interaction_id or source
    created_at: datetime
    updated_at: datetime
    active: bool = True


class FeedbackPatternRecord(BaseModel):
    """Detected pattern in user feedback"""
    pattern_id: str
    pattern_type: str  # query_refinement, feature_preference, etc.
    pattern_data: Dict[str, Any]
    frequency: int
    confidence: float
    user_ids: List[str]  # Users exhibiting this pattern
    first_seen: datetime
    last_seen: datetime


class QueryRefinementPattern(BaseModel):
    """Pattern of how users refine queries"""
    original_query: str  # Hashed
    refined_query: str  # Hashed
    refinement_type: str  # add_constraint, relax_constraint, change_focus
    frequency: int
    success_rate: float  # % of refinements that led to engagement


class PropertyPreference(BaseModel):
    """User preference for property features"""
    user_id: str
    feature_type: str  # bedrooms, location, price, style, etc.
    preferred_values: List[Any]
    weight: float = Field(ge=0.0, le=1.0)  # Importance weight
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_interactions: List[str]  # Interaction IDs
    created_at: datetime
    updated_at: datetime


class CommunicationPreference(BaseModel):
    """User preference for communication style"""
    user_id: str
    detail_level: str  # concise, balanced, detailed
    technical_level: str  # layman, intermediate, expert
    response_format: str  # bullet_points, narrative, mixed
    confidence: float = Field(ge=0.0, le=1.0)
    updated_at: datetime


# Request/Response Models for API

class FeedbackRequest(BaseModel):
    """API request for submitting feedback"""
    session_id: str
    user_id: Optional[str] = None
    feedback_type: FeedbackTypeEnum
    related_query_id: Optional[str] = None
    related_item_id: Optional[str] = None
    rating: Optional[float] = Field(None, ge=1.0, le=5.0)
    comment: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_123",
                "user_id": "user_456",
                "feedback_type": "thumbs_up",
                "related_query_id": "query_789",
                "rating": 5.0
            }
        }


class FeedbackResponse(BaseModel):
    """API response for feedback submission"""
    feedback_id: str
    status: str
    message: str


class InteractionHistoryRequest(BaseModel):
    """API request for interaction history"""
    user_id: str
    days: int = Field(default=30, ge=1, le=90)
    interaction_types: Optional[List[InteractionTypeEnum]] = None


class InteractionHistoryResponse(BaseModel):
    """API response for interaction history"""
    user_id: str
    period_days: int
    total_interactions: int
    interactions: List[InteractionRecord]


class EngagementStatsResponse(BaseModel):
    """API response for engagement statistics"""
    user_id: str
    period_days: int
    total_sessions: int
    total_queries: int
    total_feedback: int
    average_session_duration: float
    average_queries_per_session: float
    engagement_score: float  # 0-1
    top_preferences: List[Dict[str, Any]]
