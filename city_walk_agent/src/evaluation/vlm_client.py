"""
VLM API client for Qwen VLM

Handles communication with Qwen Vision Language Model API
"""

import asyncio
import base64
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from src.config import DEFAULT_RETRY_ATTEMPTS, DEFAULT_VLM_TIMEOUT
from src.utils.logging import get_logger


@dataclass
class VLMConfig:
    """Configuration for Qwen VLM"""

    api_key: str
    model: str
    api_url: str
    max_tokens: int = 300
    temperature: float = 0.7
    rate_limit_delay: float = 0.5  # seconds between requests
    max_retries: int = DEFAULT_RETRY_ATTEMPTS
    retry_delay: float = 1.0


@dataclass
class VLMStats:
    """API usage statistics"""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_time: float = 0.0
    total_cost: float = 0.0
    last_call_time: float = 0.0

    @property
    def avg_time(self) -> float:
        return self.total_time / self.total_calls if self.total_calls > 0 else 0.0

    @property
    def success_rate(self) -> float:
        return self.successful_calls / self.total_calls if self.total_calls > 0 else 0.0


class VLMClient:
    """
    Qwen VLM client

    Features:
    - Automatic rate limiting
    - Retry logic with exponential backoff
    - Cost tracking
    - Error handling
    """

    def __init__(self, config: VLMConfig) -> None:
        """
        Initialize VLM client

        Args:
            config: VLM configuration
        """
        self.config = config
        self.stats = VLMStats()
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64

        Args:
            image_path: Path to image file

        Returns:
            Base64-encoded image string
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    async def call_vlm_async(
        self, prompt: str, image_path: str, **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Async call to VLM API

        Args:
            prompt: Text prompt for evaluation
            image_path: Path to street view image
            **kwargs: Additional parameters

        Returns:
            Response dictionary with 'content' field, or None on failure
        """
        # Rate limiting
        if self.stats.last_call_time > 0:
            elapsed = time.time() - self.stats.last_call_time
            if elapsed < self.config.rate_limit_delay:
                await asyncio.sleep(self.config.rate_limit_delay - elapsed)

        # Try with retries
        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()
                response = await self._call_qwen(prompt, image_path, **kwargs)

                elapsed = time.time() - start_time
                self.stats.last_call_time = time.time()
                self.stats.total_calls += 1
                self.stats.successful_calls += 1
                self.stats.total_time += elapsed

                return response

            except Exception as error:
                self.logger.warning(
                    "VLM call attempt failed",
                    attempt=attempt + 1,
                    max_retries=self.config.max_retries,
                    error=str(error),
                )

                if attempt == self.config.max_retries - 1:
                    self.stats.total_calls += 1
                    self.stats.failed_calls += 1
                    self.logger.error(
                        "VLM call exhausted retries",
                        max_retries=self.config.max_retries,
                        error=str(error),
                    )
                    return None

                await asyncio.sleep(self.config.retry_delay * (2**attempt))

        return None

    def call_vlm(
        self, prompt: str, image_path: str, **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Synchronous wrapper for VLM call

        Args:
            prompt: Text prompt for evaluation
            image_path: Path to street view image
            **kwargs: Additional parameters

        Returns:
            Response dictionary with 'content' field, or None on failure
        """
        return asyncio.run(self.call_vlm_async(prompt, image_path, **kwargs))

    async def _call_qwen(
        self, prompt: str, image_path: str, **kwargs
    ) -> Dict[str, Any]:
        """Call Qwen VLM API"""
        image_base64 = self.encode_image(image_path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        # Use asyncio to run requests in thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(
                f"{self.config.api_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=DEFAULT_VLM_TIMEOUT,
            ),
        )

        response.raise_for_status()
        result = response.json()

        return {
            "content": result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", ""),
            "usage": result.get("usage", {}),
        }

    async def call_vlm_multi_image_async(
        self, prompt: str, image_paths: List[str], **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Call VLM with multiple images for comparison.

        Args:
            prompt: Evaluation prompt
            image_paths: List of 2-3 image paths (previous + current)
            **kwargs: Additional parameters

        Returns:
            VLM response dict or None on failure
        """
        if not image_paths or len(image_paths) > 3:
            self.logger.error(
                "Invalid image count for multi-image",
                count=len(image_paths) if image_paths else 0,
            )
            return None

        # Rate limiting
        if self.stats.last_call_time > 0:
            elapsed = time.time() - self.stats.last_call_time
            if elapsed < self.config.rate_limit_delay:
                await asyncio.sleep(self.config.rate_limit_delay - elapsed)

        # Try with retries
        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()
                response = await self._call_qwen_multi_image(
                    prompt, image_paths, **kwargs
                )

                elapsed = time.time() - start_time
                self.stats.last_call_time = time.time()
                self.stats.total_calls += 1
                self.stats.successful_calls += 1
                self.stats.total_time += elapsed

                return response

            except Exception as error:
                self.logger.warning(
                    "Multi-image VLM call attempt failed",
                    attempt=attempt + 1,
                    max_retries=self.config.max_retries,
                    error=str(error),
                )

                if attempt == self.config.max_retries - 1:
                    self.stats.total_calls += 1
                    self.stats.failed_calls += 1
                    self.logger.error(
                        "Multi-image VLM call exhausted retries",
                        max_retries=self.config.max_retries,
                        error=str(error),
                    )
                    return None

                await asyncio.sleep(self.config.retry_delay * (2**attempt))

        return None

    def call_vlm_multi_image(
        self, prompt: str, image_paths: List[str], **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Synchronous wrapper for multi-image VLM call

        Args:
            prompt: Evaluation prompt
            image_paths: List of 2-3 image paths
            **kwargs: Additional parameters

        Returns:
            Response dictionary with 'content' field, or None on failure
        """
        return asyncio.run(
            self.call_vlm_multi_image_async(prompt, image_paths, **kwargs)
        )

    async def _call_qwen_multi_image(
        self, prompt: str, image_paths: List[str], **kwargs
    ) -> Dict[str, Any]:
        """Call Qwen VLM API with multiple images"""
        # Encode all images
        images_b64 = []
        for img_path in image_paths:
            try:
                img_b64 = self.encode_image(img_path)
                images_b64.append(img_b64)
            except Exception as e:
                self.logger.error(f"Failed to encode {img_path}: {e}")
                raise

        # Build multi-image payload for Qwen
        content = [{"type": "text", "text": prompt}]
        for img_b64 in images_b64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                }
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        # Use asyncio to run requests in thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(
                f"{self.config.api_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=DEFAULT_VLM_TIMEOUT,
            ),
        )

        response.raise_for_status()
        result = response.json()

        return {
            "content": result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", ""),
            "usage": result.get("usage", {}),
            "multi_image": True,
            "image_count": len(image_paths),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        return {
            "total_calls": self.stats.total_calls,
            "successful_calls": self.stats.successful_calls,
            "failed_calls": self.stats.failed_calls,
            "success_rate": self.stats.success_rate,
            "total_time": self.stats.total_time,
            "avg_time": self.stats.avg_time,
            "total_cost": self.stats.total_cost,
        }
