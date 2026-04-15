"""Shared LLM client for System 2 components."""

import json
import os
from typing import Any, Dict, Optional

import httpx

from src.core import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


def call_llm(
    prompt: str,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 512,
) -> Optional[Dict[str, Any]]:
    """Text-only LLM call → JSON 파싱 반환. 실패 시 None.

    Args:
        prompt: 전달할 프롬프트
        api_url: API 엔드포인트 (None이면 settings에서 로드)
        api_key: API 키 (None이면 settings에서 로드)
        model: 모델명 (None이면 settings에서 로드)
        max_tokens: 최대 토큰 수

    Returns:
        파싱된 JSON dict, 실패 시 None
    """
    # Resolution order:
    # 1) explicit function args
    # 2) LLM_API_* env vars (S2 text reasoning default)
    # 3) legacy settings.vlm_* fallback for backward compatibility
    url = (
        api_url
        or os.getenv("LLM_API_URL")
        or getattr(settings, "vlm_api_url", None)
    )
    key = (
        api_key
        or os.getenv("LLM_API_KEY")
        or getattr(settings, "vlm_api_key", None)
    )
    mdl = (
        model
        or os.getenv("LLM_MODEL")
        or getattr(settings, "vlm_model", None)
    )

    if not url or not key:
        logger.warning("LLM credentials not found (checked args, LLM_API_*, then VLM_* fallback)")
        return None

    # /v1/chat/completions 엔드포인트로 변환
    if url and not url.endswith("/chat/completions"):
        url = url.rstrip("/") + "/chat/completions"

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": mdl,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            clean = content.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            return json.loads(clean.strip())
    except Exception as e:
        logger.warning(f"call_llm failed: {e}")
        return None
