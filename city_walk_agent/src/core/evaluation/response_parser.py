"""
Response parser for VLM evaluation results

Handles parsing and validation of VLM JSON responses with fallback mechanisms
"""

import json
import re
from typing import Dict, Optional, Any


class ResponseParser:
    """
    Parse VLM responses to extract scores and reasoning

    Handles:
    - JSON parsing with multiple fallback strategies
    - Score validation and normalization
    - Reasoning extraction
    - Error handling for malformed responses
    """

    @staticmethod
    def parse_response(response_text: str, dimension_id: str = "") -> Optional[Dict[str, Any]]:
        """
        Parse VLM response to extract score and reasoning

        Args:
            response_text: Raw text response from VLM
            dimension_id: Dimension being evaluated (for logging)

        Returns:
            Dict with 'score' (float) and 'reasoning' (str), or None on failure
        """
        if not response_text:
            return None

        # Clean response (remove markdown code blocks)
        cleaned_text = ResponseParser._clean_response(response_text)

        # Try JSON parsing first
        parsed = ResponseParser._try_json_parse(cleaned_text)
        if parsed:
            return parsed

        # Fallback: try to extract score and reasoning from text
        parsed = ResponseParser._try_text_extraction(cleaned_text)
        if parsed:
            return parsed

        # Last resort: return default score with full text as reasoning
        return {
            "score": 5.0,
            "reasoning": response_text[:500]  # Truncate long responses
        }

    @staticmethod
    def _clean_response(text: str) -> str:
        """
        Clean response text by removing markdown code blocks

        Args:
            text: Raw response text

        Returns:
            Cleaned text
        """
        text = text.strip()

        # Remove JSON code blocks
        if text.startswith("```json"):
            text = text.split("```json", 1)[1]
        if text.startswith("```"):
            text = text.split("```", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]

        return text.strip()

    @staticmethod
    def _try_json_parse(text: str) -> Optional[Dict[str, Any]]:
        """
        Try to parse text as JSON

        Args:
            text: Cleaned text

        Returns:
            Parsed dict with score and reasoning, or None
        """
        try:
            data = json.loads(text)

            # Extract score
            score = None
            for key in ["score", "Score", "评分", "分数"]:
                if key in data:
                    score = float(data[key])
                    break

            if score is None:
                return None

            # Validate score range
            if not (1.0 <= score <= 10.0):
                score = max(1.0, min(10.0, score))  # Clamp to valid range

            # Extract reasoning
            reasoning = ""
            for key in ["reasoning", "Reasoning", "reason", "explanation", "解释", "理由"]:
                if key in data:
                    reasoning = str(data[key])
                    break

            if not reasoning:
                reasoning = text

            return {
                "score": score,
                "reasoning": reasoning
            }

        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    @staticmethod
    def _try_text_extraction(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract score from unstructured text

        Patterns:
        - "score: 7.5"
        - "7.5/10"
        - "评分: 8"
        - "8分"

        Args:
            text: Response text

        Returns:
            Parsed dict with score and reasoning, or None
        """
        # Score patterns (English and Chinese)
        patterns = [
            r"(?:score|评分|分数)[:：]\s*(\d+(?:\.\d+)?)",  # score: 7.5
            r"(\d+(?:\.\d+)?)\s*[/／]\s*10",  # 7.5/10
            r"(\d+(?:\.\d+)?)\s*分",  # 8分
            r"(?:rate|rating)[:：]\s*(\d+(?:\.\d+)?)",  # rate: 7
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    if 1.0 <= score <= 10.0:
                        return {
                            "score": score,
                            "reasoning": text
                        }
                except ValueError:
                    continue

        return None

    @staticmethod
    def validate_evaluation_result(
        result: Dict[str, Any],
        dimension_id: str
    ) -> Dict[str, Any]:
        """
        Validate and normalize evaluation result

        Args:
            result: Parsed result dict
            dimension_id: Dimension being evaluated

        Returns:
            Validated and normalized result
        """
        validated = result.copy()

        # Validate score
        if "score" not in validated:
            validated["score"] = 5.0

        validated["score"] = float(validated["score"])

        # Clamp score to valid range
        if validated["score"] < 1.0 or validated["score"] > 10.0:
            original_score = validated["score"]
            validated["score"] = max(1.0, min(10.0, validated["score"]))
            validated["reasoning"] = (
                f"[Score adjusted from {original_score} to {validated['score']}] "
                + validated.get("reasoning", "")
            )

        # Validate reasoning
        if "reasoning" not in validated or not validated["reasoning"]:
            validated["reasoning"] = f"Score: {validated['score']}/10"

        # Truncate very long reasoning
        if len(validated["reasoning"]) > 1000:
            validated["reasoning"] = validated["reasoning"][:997] + "..."

        # Add metadata
        validated["dimension_id"] = dimension_id
        validated["parsed_successfully"] = result.get("score") is not None

        return validated

    @staticmethod
    def batch_parse_responses(
        responses: list[tuple[str, str, str]]
    ) -> list[Dict[str, Any]]:
        """
        Parse multiple responses in batch

        Args:
            responses: List of (response_text, dimension_id, image_id) tuples

        Returns:
            List of parsed and validated results
        """
        results = []

        for response_text, dimension_id, image_id in responses:
            parsed = ResponseParser.parse_response(response_text, dimension_id)

            if parsed:
                validated = ResponseParser.validate_evaluation_result(parsed, dimension_id)
                validated["image_id"] = image_id
                results.append(validated)
            else:
                # Fallback result
                results.append({
                    "score": 5.0,
                    "reasoning": "Failed to parse response",
                    "dimension_id": dimension_id,
                    "image_id": image_id,
                    "parsed_successfully": False
                })

        return results
