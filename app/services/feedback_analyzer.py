"""Feedback analysis pipeline for detecting patterns and learning preferences"""
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass

from app.services.interaction_logger import interaction_logger


@dataclass
class FeedbackPattern:
    """Detected pattern in feedback"""
    pattern_type: str
    pattern_data: Dict[str, Any]
    frequency: int
    confidence: float
    users_affected: List[str]


@dataclass
class QueryRefinement:
    """Query refinement pattern"""
    original_intent: Dict[str, Any]
    refined_intent: Dict[str, Any]
    refinement_type: str
    success_rate: float


class FeedbackAnalyzer:
    """Analyzes user interactions to detect patterns and extract preferences"""

    def __init__(self):
        """Initialize feedback analyzer"""
        self.interaction_logger = interaction_logger

    def analyze_user_patterns(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Analyze interaction patterns for a user

        Args:
            user_id: User identifier
            days: Days to analyze

        Returns:
            Pattern analysis results
        """
        # Get user history
        interactions = self.interaction_logger.get_user_history(user_id, days)

        if not interactions:
            return {
                "user_id": user_id,
                "status": "no_data",
                "message": "No interaction data found"
            }

        # Analyze different pattern types
        query_patterns = self._analyze_query_patterns(interactions)
        feedback_patterns = self._analyze_feedback_patterns(interactions)
        engagement_patterns = self._analyze_engagement(interactions)
        preference_signals = self._extract_preference_signals(interactions)

        return {
            "user_id": user_id,
            "analysis_period_days": days,
            "total_interactions": len(interactions),
            "query_patterns": query_patterns,
            "feedback_patterns": feedback_patterns,
            "engagement_patterns": engagement_patterns,
            "preference_signals": preference_signals,
            "analyzed_at": datetime.utcnow().isoformat()
        }

    def detect_query_refinement_patterns(
        self,
        min_frequency: int = 3
    ) -> List[QueryRefinement]:
        """Detect common query refinement patterns across all users

        Args:
            min_frequency: Minimum times pattern must occur

        Returns:
            List of detected refinement patterns
        """
        refinement_patterns = []

        # Would analyze query pairs to find common refinements
        # Simplified implementation

        return refinement_patterns

    def classify_interaction_success(
        self,
        interaction: Dict[str, Any]
    ) -> str:
        """Classify if interaction was successful

        Args:
            interaction: Interaction data

        Returns:
            Classification (success, failure, neutral)
        """
        interaction_type = interaction.get("interaction_type")
        data = interaction.get("data", {})

        # Query success indicators
        if interaction_type == "query":
            results_count = data.get("results_count", 0)
            if results_count == 0:
                return "failure"
            elif results_count >= 5:
                return "success"
            else:
                return "neutral"

        # Feedback success indicators
        elif interaction_type == "feedback":
            feedback_type = data.get("feedback_type", "")
            if feedback_type in ["thumbs_up", "property_saved", "property_shared"]:
                return "success"
            elif feedback_type == "thumbs_down":
                return "failure"

        # Error is always failure
        elif interaction_type == "error":
            return "failure"

        return "neutral"

    def extract_user_preferences(
        self,
        user_id: str,
        min_confidence: float = 0.6
    ) -> List[Dict[str, Any]]:
        """Extract learned preferences for a user

        Args:
            user_id: User identifier
            min_confidence: Minimum confidence threshold

        Returns:
            List of preferences with confidence scores
        """
        interactions = self.interaction_logger.get_user_history(user_id, days=90)

        if not interactions:
            return []

        preferences = []

        # Extract property feature preferences
        feature_prefs = self._extract_feature_preferences(interactions)
        preferences.extend([p for p in feature_prefs if p["confidence"] >= min_confidence])

        # Extract location preferences
        location_prefs = self._extract_location_preferences(interactions)
        preferences.extend([p for p in location_prefs if p["confidence"] >= min_confidence])

        # Extract price range preferences
        price_prefs = self._extract_price_preferences(interactions)
        preferences.extend([p for p in price_prefs if p["confidence"] >= min_confidence])

        # Sort by confidence
        preferences.sort(key=lambda p: p["confidence"], reverse=True)

        return preferences

    def _analyze_query_patterns(
        self,
        interactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze query interaction patterns

        Args:
            interactions: User interactions

        Returns:
            Query pattern analysis
        """
        queries = [i for i in interactions if i.get("interaction_type") == "query"]

        if not queries:
            return {"total_queries": 0}

        # Analyze search modes
        search_modes = Counter([q.get("data", {}).get("search_mode") for q in queries])

        # Analyze query times
        query_times = [datetime.fromisoformat(q.get("timestamp", "")).hour for q in queries]
        peak_hours = Counter(query_times).most_common(3)

        # Analyze result counts
        result_counts = [q.get("data", {}).get("results_count", 0) for q in queries]
        avg_results = sum(result_counts) / len(result_counts) if result_counts else 0

        return {
            "total_queries": len(queries),
            "search_modes": dict(search_modes),
            "peak_hours": [{"hour": h, "count": c} for h, c in peak_hours],
            "average_results": avg_results,
            "zero_result_rate": sum(1 for r in result_counts if r == 0) / len(result_counts) if result_counts else 0
        }

    def _analyze_feedback_patterns(
        self,
        interactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze feedback interaction patterns

        Args:
            interactions: User interactions

        Returns:
            Feedback pattern analysis
        """
        feedbacks = [i for i in interactions if i.get("interaction_type") == "feedback"]

        if not feedbacks:
            return {"total_feedback": 0}

        # Analyze feedback types
        feedback_types = Counter([f.get("data", {}).get("feedback_type") for f in feedbacks])

        # Calculate positive feedback rate
        positive_types = ["thumbs_up", "property_saved", "property_shared"]
        positive_count = sum(feedback_types.get(t, 0) for t in positive_types)
        positive_rate = positive_count / len(feedbacks) if feedbacks else 0

        return {
            "total_feedback": len(feedbacks),
            "feedback_types": dict(feedback_types),
            "positive_feedback_rate": positive_rate,
            "engagement_score": positive_rate  # Simplified engagement score
        }

    def _analyze_engagement(
        self,
        interactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze user engagement

        Args:
            interactions: User interactions

        Returns:
            Engagement analysis
        """
        # Count sessions
        sessions = set(i.get("session_id") for i in interactions)

        # Count interaction types
        queries = sum(1 for i in interactions if i.get("interaction_type") == "query")
        feedbacks = sum(1 for i in interactions if i.get("interaction_type") == "feedback")
        clicks = sum(1 for i in interactions if i.get("interaction_type") == "click")

        # Calculate engagement score (0-1)
        # Based on query:feedback ratio, click rate, etc.
        engagement_score = min(1.0, (feedbacks / queries if queries > 0 else 0) * 0.5 + (clicks / queries if queries > 0 else 0) * 0.5)

        return {
            "total_sessions": len(sessions),
            "total_interactions": len(interactions),
            "queries": queries,
            "feedbacks": feedbacks,
            "clicks": clicks,
            "engagement_score": engagement_score,
            "feedback_rate": feedbacks / queries if queries > 0 else 0
        }

    def _extract_preference_signals(
        self,
        interactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract preference signals from interactions

        Args:
            interactions: User interactions

        Returns:
            Preference signals
        """
        signals = {
            "property_features": defaultdict(int),
            "locations": defaultdict(int),
            "price_ranges": defaultdict(int),
            "search_preferences": defaultdict(int)
        }

        # Extract from query intents
        queries = [i for i in interactions if i.get("interaction_type") == "query"]
        for query in queries:
            intent = query.get("data", {}).get("intent_extracted", {})

            # Property features
            if "bedrooms" in intent:
                signals["property_features"][f"bedrooms_{intent['bedrooms'].get('min', 'any')}"] += 1

            # Locations
            if "location" in intent:
                location = intent["location"]
                if isinstance(location, dict):
                    city = location.get("city", "")
                    if city:
                        signals["locations"][city] += 1
                elif isinstance(location, str):
                    signals["locations"][location] += 1

            # Price ranges
            if "price_range" in intent:
                price_range = intent["price_range"]
                if isinstance(price_range, dict):
                    max_price = price_range.get("max", 0)
                    if max_price:
                        # Bucket into ranges
                        if max_price < 500000:
                            signals["price_ranges"]["under_500k"] += 1
                        elif max_price < 1000000:
                            signals["price_ranges"]["500k_1m"] += 1
                        else:
                            signals["price_ranges"]["over_1m"] += 1

        # Convert to list of top preferences
        return {
            "top_features": dict(Counter(signals["property_features"]).most_common(5)),
            "top_locations": dict(Counter(signals["locations"]).most_common(5)),
            "top_price_ranges": dict(Counter(signals["price_ranges"]).most_common(3)),
            "search_mode_preference": dict(signals["search_preferences"])
        }

    def _extract_feature_preferences(
        self,
        interactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract property feature preferences"""
        preferences = []

        # Analyze bedrooms preference
        bedroom_counts = defaultdict(int)
        queries = [i for i in interactions if i.get("interaction_type") == "query"]

        for query in queries:
            intent = query.get("data", {}).get("intent_extracted", {})
            if "bedrooms" in intent:
                bedrooms = intent["bedrooms"]
                if isinstance(bedrooms, dict):
                    min_br = bedrooms.get("min", bedrooms.get("preferred"))
                    if min_br:
                        bedroom_counts[min_br] += 1

        if bedroom_counts:
            most_common = max(bedroom_counts.items(), key=lambda x: x[1])
            confidence = most_common[1] / sum(bedroom_counts.values())

            if confidence >= 0.5:  # Appears in >50% of queries
                preferences.append({
                    "type": "property_feature",
                    "feature": "bedrooms",
                    "value": most_common[0],
                    "confidence": confidence,
                    "frequency": most_common[1]
                })

        return preferences

    def _extract_location_preferences(
        self,
        interactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract location preferences"""
        preferences = []

        location_counts = defaultdict(int)
        queries = [i for i in interactions if i.get("interaction_type") == "query"]

        for query in queries:
            intent = query.get("data", {}).get("intent_extracted", {})
            if "location" in intent:
                loc = intent["location"]
                if isinstance(loc, str):
                    location_counts[loc] += 1
                elif isinstance(loc, dict):
                    city = loc.get("city", "")
                    if city:
                        location_counts[city] += 1

        # Top 3 locations
        for location, count in Counter(location_counts).most_common(3):
            confidence = count / sum(location_counts.values()) if location_counts else 0
            if confidence >= 0.3:  # Appears in >30% of queries
                preferences.append({
                    "type": "location",
                    "feature": "city",
                    "value": location,
                    "confidence": confidence,
                    "frequency": count
                })

        return preferences

    def _extract_price_preferences(
        self,
        interactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract price range preferences"""
        preferences = []

        price_ranges = []
        queries = [i for i in interactions if i.get("interaction_type") == "query"]

        for query in queries:
            intent = query.get("data", {}).get("intent_extracted", {})
            if "price_range" in intent:
                price_range = intent["price_range"]
                if isinstance(price_range, dict):
                    max_price = price_range.get("max", 0)
                    if max_price:
                        price_ranges.append(max_price)

        if price_ranges:
            avg_max = sum(price_ranges) / len(price_ranges)
            std_dev = (sum((x - avg_max) ** 2 for x in price_ranges) / len(price_ranges)) ** 0.5
            consistency = 1.0 - min(1.0, std_dev / avg_max) if avg_max > 0 else 0

            preferences.append({
                "type": "price_range",
                "feature": "max_price",
                "value": avg_max,
                "confidence": consistency,
                "frequency": len(price_ranges)
            })

        return preferences


# Global instance
feedback_analyzer = FeedbackAnalyzer()
