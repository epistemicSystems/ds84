"""Preference learning system for personalizing search and recommendations"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from app.services.feedback_analyzer import feedback_analyzer
from app.services.interaction_logger import interaction_logger


class PreferenceLearner:
    """Learns and applies user preferences for personalization"""

    def __init__(self):
        """Initialize preference learner"""
        self.feedback_analyzer = feedback_analyzer
        self.interaction_logger = interaction_logger

        # User preference cache
        self.user_preferences: Dict[str, Dict[str, Any]] = {}

    def learn_preferences(
        self,
        user_id: str,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """Learn preferences for a user

        Args:
            user_id: User identifier
            force_refresh: Force refresh from interactions

        Returns:
            Learned preferences
        """
        # Check cache
        if not force_refresh and user_id in self.user_preferences:
            cached = self.user_preferences[user_id]
            # Check if cache is fresh (< 1 hour old)
            if (datetime.utcnow() - cached.get("learned_at", datetime.min)).total_seconds() < 3600:
                return cached

        # Extract preferences from interactions
        preferences = self.feedback_analyzer.extract_user_preferences(user_id, min_confidence=0.5)

        # Organize preferences by category
        organized = {
            "property_features": [],
            "locations": [],
            "price_range": {},
            "communication_style": {},
            "search_behavior": {},
            "learned_at": datetime.utcnow()
        }

        for pref in preferences:
            pref_type = pref.get("type")

            if pref_type == "property_feature":
                organized["property_features"].append({
                    "feature": pref.get("feature"),
                    "value": pref.get("value"),
                    "confidence": pref.get("confidence"),
                    "weight": pref.get("confidence")  # Use confidence as weight
                })

            elif pref_type == "location":
                organized["locations"].append({
                    "city": pref.get("value"),
                    "confidence": pref.get("confidence"),
                    "weight": pref.get("confidence")
                })

            elif pref_type == "price_range":
                organized["price_range"] = {
                    "max_price": pref.get("value"),
                    "confidence": pref.get("confidence")
                }

        # Infer communication style from interaction patterns
        comm_style = self._infer_communication_style(user_id)
        organized["communication_style"] = comm_style

        # Infer search behavior patterns
        search_behavior = self._infer_search_behavior(user_id)
        organized["search_behavior"] = search_behavior

        # Cache preferences
        self.user_preferences[user_id] = organized

        return organized

    def apply_preferences_to_query(
        self,
        user_id: str,
        query: str,
        intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply learned preferences to enhance query intent

        Args:
            user_id: User identifier
            query: Original query
            intent: Extracted intent

        Returns:
            Enhanced intent with preferences
        """
        # Get user preferences
        preferences = self.learn_preferences(user_id)

        # Create enhanced intent
        enhanced_intent = intent.copy()

        # Apply property feature preferences
        for feature_pref in preferences.get("property_features", []):
            feature = feature_pref.get("feature")
            value = feature_pref.get("value")
            weight = feature_pref.get("weight", 0.5)

            # Only apply if not explicitly specified in query
            if feature not in intent or intent[feature] is None:
                if weight > 0.6:  # High confidence preference
                    enhanced_intent[feature] = {
                        "preferred": value,
                        "from_preference": True,
                        "confidence": weight
                    }

        # Apply location preferences if no location specified
        if "location" not in intent or not intent["location"]:
            top_locations = preferences.get("locations", [])
            if top_locations and top_locations[0].get("confidence", 0) > 0.5:
                enhanced_intent["preferred_locations"] = [
                    loc.get("city") for loc in top_locations[:3]
                ]

        # Apply price range preference if no price specified
        if "price_range" not in intent or not intent["price_range"]:
            price_pref = preferences.get("price_range", {})
            if price_pref and price_pref.get("confidence", 0) > 0.5:
                enhanced_intent["price_range"] = {
                    "max": price_pref.get("max_price"),
                    "from_preference": True
                }

        return enhanced_intent

    def rank_results_by_preferences(
        self,
        user_id: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Re-rank search results based on user preferences

        Args:
            user_id: User identifier
            results: Search results

        Returns:
            Re-ranked results
        """
        # Get user preferences
        preferences = self.learn_preferences(user_id)

        # Score each result
        scored_results = []
        for result in results:
            score = self._calculate_preference_score(result, preferences)
            scored_results.append({
                **result,
                "preference_score": score,
                "original_rank": results.index(result)
            })

        # Sort by preference score (descending), then original rank
        scored_results.sort(
            key=lambda r: (r["preference_score"], -r["original_rank"]),
            reverse=True
        )

        return scored_results

    def explain_preferences(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """Generate explanation of learned preferences

        Args:
            user_id: User identifier

        Returns:
            Preference explanation
        """
        preferences = self.learn_preferences(user_id)

        explanations = {
            "summary": [],
            "details": {},
            "confidence_level": "high"  # high, medium, low
        }

        # Explain property features
        features = preferences.get("property_features", [])
        if features:
            high_conf_features = [f for f in features if f.get("confidence", 0) > 0.7]
            if high_conf_features:
                feature_list = [f"{f['feature']}: {f['value']}" for f in high_conf_features]
                explanations["summary"].append(
                    f"You typically search for properties with {', '.join(feature_list)}"
                )
                explanations["details"]["property_features"] = high_conf_features

        # Explain location preferences
        locations = preferences.get("locations", [])
        if locations:
            top_locs = [loc.get("city") for loc in locations[:3] if loc.get("confidence", 0) > 0.5]
            if top_locs:
                explanations["summary"].append(
                    f"Your preferred locations are: {', '.join(top_locs)}"
                )
                explanations["details"]["locations"] = locations[:3]

        # Explain price range
        price_range = preferences.get("price_range", {})
        if price_range and price_range.get("confidence", 0) > 0.5:
            max_price = price_range.get("max_price", 0)
            explanations["summary"].append(
                f"Your typical price range is up to ${max_price:,.0f}"
            )
            explanations["details"]["price_range"] = price_range

        # Explain communication style
        comm_style = preferences.get("communication_style", {})
        if comm_style:
            detail_level = comm_style.get("detail_level", "balanced")
            explanations["summary"].append(
                f"You prefer {detail_level} responses"
            )
            explanations["details"]["communication_style"] = comm_style

        # Determine overall confidence
        all_confidences = []
        for feature in features:
            all_confidences.append(feature.get("confidence", 0))
        for loc in locations:
            all_confidences.append(loc.get("confidence", 0))
        if price_range:
            all_confidences.append(price_range.get("confidence", 0))

        if all_confidences:
            avg_confidence = sum(all_confidences) / len(all_confidences)
            if avg_confidence > 0.7:
                explanations["confidence_level"] = "high"
            elif avg_confidence > 0.5:
                explanations["confidence_level"] = "medium"
            else:
                explanations["confidence_level"] = "low"

        return explanations

    def _infer_communication_style(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """Infer preferred communication style

        Args:
            user_id: User identifier

        Returns:
            Communication style preferences
        """
        interactions = self.interaction_logger.get_user_history(user_id, days=30)

        # Analyze query complexity
        queries = [i for i in interactions if i.get("interaction_type") == "query"]

        if not queries:
            return {
                "detail_level": "balanced",
                "technical_level": "intermediate",
                "response_format": "mixed",
                "confidence": 0.5
            }

        # Simple heuristic: longer queries suggest preference for detailed responses
        avg_query_length = sum(len(str(q.get("data", {}).get("query_text", ""))) for q in queries) / len(queries)

        if avg_query_length > 100:
            detail_level = "detailed"
        elif avg_query_length < 50:
            detail_level = "concise"
        else:
            detail_level = "balanced"

        return {
            "detail_level": detail_level,
            "technical_level": "intermediate",  # Could be inferred from terminology used
            "response_format": "mixed",
            "confidence": 0.6
        }

    def _infer_search_behavior(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """Infer search behavior patterns

        Args:
            user_id: User identifier

        Returns:
            Search behavior patterns
        """
        interactions = self.interaction_logger.get_user_history(user_id, days=30)

        queries = [i for i in interactions if i.get("interaction_type") == "query"]
        refinements = [i for i in interactions if i.get("interaction_type") == "refinement"]

        return {
            "average_queries_per_session": len(queries) / max(1, len(set(i.get("session_id") for i in interactions))),
            "refinement_rate": len(refinements) / len(queries) if queries else 0,
            "search_mode_preference": self._get_preferred_search_mode(queries),
            "time_of_day_preference": self._get_preferred_time(queries)
        }

    def _get_preferred_search_mode(
        self,
        queries: List[Dict[str, Any]]
    ) -> str:
        """Get preferred search mode from queries"""
        if not queries:
            return "vector"

        modes = [q.get("data", {}).get("search_mode", "vector") for q in queries]
        from collections import Counter
        most_common = Counter(modes).most_common(1)

        return most_common[0][0] if most_common else "vector"

    def _get_preferred_time(
        self,
        queries: List[Dict[str, Any]]
    ) -> str:
        """Get preferred time of day for searches"""
        if not queries:
            return "afternoon"

        hours = [datetime.fromisoformat(q.get("timestamp", "")).hour for q in queries]
        avg_hour = sum(hours) / len(hours) if hours else 14

        if avg_hour < 6:
            return "night"
        elif avg_hour < 12:
            return "morning"
        elif avg_hour < 18:
            return "afternoon"
        else:
            return "evening"

    def _calculate_preference_score(
        self,
        result: Dict[str, Any],
        preferences: Dict[str, Any]
    ) -> float:
        """Calculate how well a result matches user preferences

        Args:
            result: Search result
            preferences: User preferences

        Returns:
            Preference match score (0-1)
        """
        score = 0.0
        max_score = 0.0

        # Check property features
        for feature_pref in preferences.get("property_features", []):
            feature = feature_pref.get("feature")
            value = feature_pref.get("value")
            weight = feature_pref.get("weight", 0.5)

            max_score += weight

            if feature in result:
                result_value = result.get(feature)
                if result_value == value:
                    score += weight
                elif isinstance(result_value, (int, float)) and isinstance(value, (int, float)):
                    # Partial credit for close matches
                    diff = abs(result_value - value) / max(result_value, value)
                    score += weight * max(0, 1 - diff)

        # Check location
        for loc_pref in preferences.get("locations", []):
            city = loc_pref.get("city", "")
            weight = loc_pref.get("weight", 0.5)

            max_score += weight

            result_location = result.get("address", {}).get("city", "")
            if city.lower() in result_location.lower():
                score += weight

        # Check price range
        price_pref = preferences.get("price_range", {})
        if price_pref:
            max_price = price_pref.get("max_price", 0)
            confidence = price_pref.get("confidence", 0.5)

            max_score += confidence

            result_price = result.get("price", 0)
            if result_price <= max_price:
                # More score for prices closer to preference
                score += confidence * (1 - (result_price / max_price if max_price > 0 else 0) * 0.5)

        # Normalize score
        return score / max_score if max_score > 0 else 0.5


# Global instance
preference_learner = PreferenceLearner()
