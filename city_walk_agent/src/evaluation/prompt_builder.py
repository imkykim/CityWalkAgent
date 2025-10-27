"""
Dynamic prompt builder for VLM evaluation

Generates evaluation prompts from framework JSON configurations
"""

from typing import Dict, List, Optional, Any


class PromptBuilder:
    """
    Build evaluation prompts dynamically from framework configurations

    Features:
    - Dynamic prompt generation from JSON framework definitions
    - Multi-language support (English/Chinese)
    - Framework-specific customization
    - Batch prompt generation
    """

    def __init__(self, framework: Dict[str, Any]):
        """
        Initialize prompt builder with framework configuration

        Args:
            framework: Framework configuration dict (loaded from JSON)
        """
        self.framework = framework
        self.framework_id = framework["framework_id"]
        self.framework_name = framework.get("framework_name", "")
        self.dimensions = framework["dimensions"]

    def build_dimension_prompt(
        self,
        dimension_id: str,
        language: str = "cn"
    ) -> Optional[str]:
        """
        Build prompt for a specific dimension

        Args:
            dimension_id: Dimension identifier (e.g., "safety", "comfort")
            language: Language for prompt ("en" or "cn")

        Returns:
            Evaluation prompt string, or None if dimension not found
        """
        # Find dimension in framework
        dimension = None
        for dim in self.dimensions:
            if dim["id"] == dimension_id:
                dimension = dim
                break

        if not dimension:
            return None

        # Use pre-defined VLM prompt if available
        if "vlm_prompt" in dimension:
            return dimension["vlm_prompt"]

        # Otherwise, construct prompt from dimension metadata
        if language == "cn":
            return self._build_chinese_prompt(dimension)
        else:
            return self._build_english_prompt(dimension)

    def _build_chinese_prompt(self, dimension: Dict) -> str:
        """Build Chinese evaluation prompt"""
        dimension_name = dimension.get("name_cn", dimension.get("name_en", ""))
        description = dimension.get("description", "")

        prompt = f"""请评估这张街景图像的**{dimension_name}**。

{description}

请以1-10分评分（1分=最差，10分=最好），并用2-3句话解释评分理由。

请以JSON格式回复：
{{"score": <1-10的数字>, "reasoning": "<中文解释>"}}"""

        return prompt

    def _build_english_prompt(self, dimension: Dict) -> str:
        """Build English evaluation prompt"""
        dimension_name = dimension.get("name_en", dimension.get("name_cn", ""))
        description = dimension.get("description", "")

        prompt = f"""Please evaluate the **{dimension_name}** of this street view image.

{description}

Rate on a scale of 1-10 (1=worst, 10=best) and provide 2-3 sentences explaining your rating.

Please respond in JSON format:
{{"score": <number 1-10>, "reasoning": "<explanation>"}}"""

        return prompt

    def build_all_prompts(self, language: str = "cn") -> Dict[str, str]:
        """
        Build prompts for all dimensions in framework

        Args:
            language: Language for prompts ("en" or "cn")

        Returns:
            Dict mapping dimension_id to prompt string
        """
        prompts = {}

        for dimension in self.dimensions:
            dimension_id = dimension["id"]
            prompt = self.build_dimension_prompt(dimension_id, language)
            if prompt:
                prompts[dimension_id] = prompt

        return prompts

    def get_framework_context(self) -> str:
        """
        Get framework context description for logging/reporting

        Returns:
            Framework description string
        """
        return (
            f"{self.framework_name} ({self.framework_id})\n"
            f"Dimensions: {len(self.dimensions)}\n"
            f"Theory: {self.framework.get('theory_base', 'N/A')}"
        )

    @staticmethod
    def create_batch_prompts(
        frameworks: List[Dict[str, Any]],
        language: str = "cn"
    ) -> Dict[str, Dict[str, str]]:
        """
        Create prompts for multiple frameworks

        Args:
            frameworks: List of framework configuration dicts
            language: Language for prompts

        Returns:
            Dict mapping framework_id to dimension_prompts dict
        """
        batch_prompts = {}

        for framework in frameworks:
            framework_id = framework["framework_id"]
            builder = PromptBuilder(framework)
            batch_prompts[framework_id] = builder.build_all_prompts(language)

        return batch_prompts

    def get_dimension_metadata(self, dimension_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full metadata for a dimension

        Args:
            dimension_id: Dimension identifier

        Returns:
            Dimension metadata dict, or None if not found
        """
        for dimension in self.dimensions:
            if dimension["id"] == dimension_id:
                return dimension

        return None

    def get_all_dimension_ids(self) -> List[str]:
        """
        Get list of all dimension IDs in framework

        Returns:
            List of dimension IDs
        """
        return [dim["id"] for dim in self.dimensions]

    def get_framework_info(self) -> Dict[str, Any]:
        """
        Get framework information for reporting

        Returns:
            Dict with framework metadata
        """
        return {
            "framework_id": self.framework_id,
            "framework_name": self.framework_name,
            "framework_name_cn": self.framework.get("framework_name_cn", ""),
            "num_dimensions": len(self.dimensions),
            "dimensions": [
                {
                    "id": dim["id"],
                    "name_en": dim.get("name_en", ""),
                    "name_cn": dim.get("name_cn", "")
                }
                for dim in self.dimensions
            ],
            "theory_base": self.framework.get("theory_base", ""),
            "year": self.framework.get("year", None)
        }
