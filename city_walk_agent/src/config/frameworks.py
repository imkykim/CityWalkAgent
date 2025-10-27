"""
Framework management utilities

Load and manage evaluation frameworks from JSON configurations
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any


class FrameworkManager:
    """
    Manage evaluation frameworks

    Provides utilities for:
    - Loading framework configurations from JSON
    - Listing available frameworks
    - Validating framework schemas
    - Framework metadata access
    """

    def __init__(self, frameworks_dir: Optional[Path] = None):
        """
        Initialize framework manager

        Args:
            frameworks_dir: Directory containing framework JSON files
                           (defaults to src/config/framework_configs/)
        """
        if frameworks_dir is None:
            # Default to project's frameworks directory
            self.frameworks_dir = Path(__file__).parent / "framework_configs"
        else:
            self.frameworks_dir = Path(frameworks_dir)

        self._frameworks_cache: Dict[str, Dict[str, Any]] = {}

    def load_framework(self, framework_id: str) -> Dict[str, Any]:
        """
        Load framework by ID

        Args:
            framework_id: Framework identifier (e.g., "sagai_2025")

        Returns:
            Framework configuration dict

        Raises:
            FileNotFoundError: If framework file not found
            ValueError: If framework validation fails
        """
        # Check cache first
        if framework_id in self._frameworks_cache:
            return self._frameworks_cache[framework_id]

        # Load from file
        framework_path = self.frameworks_dir / f"{framework_id}.json"

        if not framework_path.exists():
            raise FileNotFoundError(
                f"Framework '{framework_id}' not found at {framework_path}"
            )

        try:
            with open(framework_path, 'r', encoding='utf-8') as f:
                framework = json.load(f)

            # Validate framework
            self._validate_framework(framework)

            # Cache and return
            self._frameworks_cache[framework_id] = framework
            return framework

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in framework file {framework_path}: {e}")

    def load_all_frameworks(self) -> List[Dict[str, Any]]:
        """
        Load all available frameworks

        Returns:
            List of framework configuration dicts
        """
        if not self.frameworks_dir.exists():
            print(f"Warning: Frameworks directory not found: {self.frameworks_dir}")
            return []

        frameworks = []
        framework_files = sorted(self.frameworks_dir.glob("*.json"))

        for file_path in framework_files:
            try:
                framework_id = file_path.stem
                framework = self.load_framework(framework_id)
                frameworks.append(framework)
            except Exception as e:
                print(f"Warning: Failed to load {file_path.name}: {e}")
                continue

        return frameworks

    def list_available_frameworks(self) -> List[Dict[str, str]]:
        """
        List available frameworks with metadata

        Returns:
            List of dicts with framework metadata (id, name, dimensions)
        """
        frameworks = self.load_all_frameworks()

        return [
            {
                "id": f["framework_id"],
                "name": f.get("framework_name", ""),
                "name_cn": f.get("framework_name_cn", ""),
                "num_dimensions": f.get("num_dimensions", len(f.get("dimensions", []))),
                "year": f.get("year", None)
            }
            for f in frameworks
        ]

    def get_framework_info(self, framework_id: str) -> Dict[str, Any]:
        """
        Get detailed framework information

        Args:
            framework_id: Framework identifier

        Returns:
            Framework metadata dict
        """
        framework = self.load_framework(framework_id)

        return {
            "framework_id": framework["framework_id"],
            "framework_name": framework.get("framework_name", ""),
            "framework_name_cn": framework.get("framework_name_cn", ""),
            "description": framework.get("description", ""),
            "description_cn": framework.get("description_cn", ""),
            "theory_base": framework.get("theory_base", ""),
            "year": framework.get("year", None),
            "num_dimensions": len(framework.get("dimensions", [])),
            "dimensions": [
                {
                    "id": dim["id"],
                    "name_en": dim.get("name_en", ""),
                    "name_cn": dim.get("name_cn", ""),
                    "description": dim.get("description", "")
                }
                for dim in framework.get("dimensions", [])
            ]
        }

    def get_dimension_names(
        self,
        framework_id: str,
        language: str = "cn"
    ) -> Dict[str, str]:
        """
        Get dimension names for framework

        Args:
            framework_id: Framework identifier
            language: Language for names ("en" or "cn")

        Returns:
            Dict mapping dimension_id to name
        """
        framework = self.load_framework(framework_id)

        name_key = "name_cn" if language == "cn" else "name_en"

        return {
            dim["id"]: dim.get(name_key, dim["id"])
            for dim in framework.get("dimensions", [])
        }

    def _validate_framework(self, framework: Dict[str, Any]) -> None:
        """
        Validate framework schema

        Args:
            framework: Framework configuration dict

        Raises:
            ValueError: If validation fails
        """
        required_fields = ["framework_id", "dimensions"]

        for field in required_fields:
            if field not in framework:
                raise ValueError(f"Framework missing required field: {field}")

        # Validate dimensions
        if not isinstance(framework["dimensions"], list):
            raise ValueError("Framework 'dimensions' must be a list")

        if len(framework["dimensions"]) == 0:
            raise ValueError("Framework must have at least one dimension")

        # Validate each dimension
        for i, dimension in enumerate(framework["dimensions"]):
            if not isinstance(dimension, dict):
                raise ValueError(f"Dimension {i} must be a dict")

            if "id" not in dimension:
                raise ValueError(f"Dimension {i} missing 'id' field")

    def framework_exists(self, framework_id: str) -> bool:
        """
        Check if framework exists

        Args:
            framework_id: Framework identifier

        Returns:
            True if framework exists
        """
        framework_path = self.frameworks_dir / f"{framework_id}.json"
        return framework_path.exists()

    def clear_cache(self) -> None:
        """Clear framework cache"""
        self._frameworks_cache.clear()


# Global framework manager instance
_framework_manager: Optional[FrameworkManager] = None


def get_framework_manager() -> FrameworkManager:
    """
    Get global framework manager instance

    Returns:
        FrameworkManager singleton
    """
    global _framework_manager

    if _framework_manager is None:
        _framework_manager = FrameworkManager()

    return _framework_manager


def load_framework(framework_id: str) -> Dict[str, Any]:
    """
    Convenience function to load framework

    Args:
        framework_id: Framework identifier

    Returns:
        Framework configuration dict
    """
    manager = get_framework_manager()
    return manager.load_framework(framework_id)


def list_frameworks() -> List[Dict[str, str]]:
    """
    Convenience function to list frameworks

    Returns:
        List of framework metadata dicts
    """
    manager = get_framework_manager()
    return manager.list_available_frameworks()
