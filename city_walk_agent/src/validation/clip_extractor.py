"""CLIP feature extractor for image similarity-based validation.

This module provides utilities to extract CLIP embeddings from images,
which are used for finding visually similar images via K-NN.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


class CLIPExtractor:
    """Extract CLIP embeddings from images."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        batch_size: int = 32,
    ):
        """Initialize CLIP model and processor.

        Args:
            model_name: HuggingFace model identifier
                       Default: "openai/clip-vit-base-patch32" (512-dim)
                       Alternative: "openai/clip-vit-large-patch14" (768-dim)
            device: Device to use ("cpu", "cuda", "mps", or "auto")
            batch_size: Number of images to process at once
        """
        self.model_name = model_name
        self.batch_size = batch_size

        # Auto-detect device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Initializing CLIP model: {model_name}")
        print(f"Using device: {self.device}")

        # Load model and processor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Get feature dimension
        self.feature_dim = self.model.config.projection_dim

        print(f"Feature dimension: {self.feature_dim}")

    def extract_single(self, image_path: Path) -> np.ndarray:
        """Extract feature vector for single image.

        Args:
            image_path: Path to image file

        Returns:
            Normalized feature vector of shape (feature_dim,)
            For base model: (512,), for large model: (768,)

        Raises:
            FileNotFoundError: If image doesn't exist
            Exception: If image cannot be processed
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Extract features
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

            # Normalize features (L2 normalization)
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )

            # Convert to numpy
            features = image_features.cpu().numpy().squeeze()

            return features

        except Exception as e:
            raise Exception(f"Error processing image {image_path}: {e}")

    def extract_batch(
        self, image_paths: List[Path], show_progress: bool = True
    ) -> np.ndarray:
        """Extract features for multiple images.

        Args:
            image_paths: List of paths to image files
            show_progress: Whether to show progress bar

        Returns:
            Normalized feature array of shape (N, feature_dim)
            For base model: (N, 512), for large model: (N, 768)
        """
        features_list = []
        n_batches = (len(image_paths) + self.batch_size - 1) // self.batch_size

        # Create progress bar
        iterator = range(0, len(image_paths), self.batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc="Extracting CLIP features",
                total=n_batches,
                unit="batch",
            )

        for i in iterator:
            batch_paths = image_paths[i : i + self.batch_size]

            # Load and preprocess batch
            images = []
            valid_indices = []

            for j, path in enumerate(batch_paths):
                try:
                    image = Image.open(path).convert("RGB")
                    images.append(image)
                    valid_indices.append(j)
                except Exception as e:
                    print(f"\nWarning: Failed to load {path}: {e}")
                    # Add zero vector as placeholder
                    features_list.append(np.zeros(self.feature_dim))

            if not images:
                continue

            # Process batch
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Extract features
            with torch.no_grad():
                batch_features = self.model.get_image_features(**inputs)

            # Normalize features
            batch_features = batch_features / batch_features.norm(
                dim=-1, keepdim=True
            )

            # Convert to numpy and add to list
            batch_features_np = batch_features.cpu().numpy()
            features_list.extend(batch_features_np)

        # Stack into single array
        features = np.vstack(features_list)

        return features

    def extract_and_cache(
        self,
        image_paths: List[Path],
        cache_path: Path,
        force_recompute: bool = False,
    ) -> np.ndarray:
        """Extract features with caching.

        If cache exists and matches configuration, load from cache.
        Otherwise, extract features and save to cache.

        Args:
            image_paths: List of paths to image files
            cache_path: Path to save/load cached features (.npy file)
            force_recompute: Force recomputation even if cache exists

        Returns:
            Normalized feature array of shape (N, feature_dim)
        """
        cache_path = Path(cache_path)
        metadata_path = cache_path.with_suffix(".json")

        # Check if cache exists and is valid
        if (
            not force_recompute
            and cache_path.exists()
            and metadata_path.exists()
        ):
            try:
                # Load metadata
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                # Validate cache
                if (
                    metadata.get("model_name") == self.model_name
                    and metadata.get("num_images") == len(image_paths)
                    and metadata.get("feature_dim") == self.feature_dim
                ):
                    print(f"Loading cached features from: {cache_path}")
                    features = np.load(cache_path)

                    # Verify shape
                    expected_shape = (len(image_paths), self.feature_dim)
                    if features.shape == expected_shape:
                        print(
                            f"✓ Loaded {features.shape[0]} cached feature vectors"
                        )
                        return features
                    else:
                        print(
                            f"Warning: Cached features shape mismatch. "
                            f"Expected {expected_shape}, got {features.shape}"
                        )

            except Exception as e:
                print(f"Warning: Failed to load cache: {e}")

        # Extract features
        print("Extracting CLIP features (this may take a while)...")
        features = self.extract_batch(image_paths)

        # Save cache
        print(f"Saving features to cache: {cache_path}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Save features
        np.save(cache_path, features)

        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "feature_dim": self.feature_dim,
            "num_images": len(image_paths),
            "extraction_date": datetime.now().isoformat(),
            "device": self.device,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Saved {features.shape[0]} feature vectors to cache")

        return features

    def verify_features(self, features: np.ndarray) -> dict:
        """Verify extracted features are valid.

        Args:
            features: Feature array to verify

        Returns:
            Dictionary with verification results
        """
        results = {
            "shape": features.shape,
            "feature_dim_matches": features.shape[1] == self.feature_dim,
            "mean_l2_norm": float(np.linalg.norm(features, axis=1).mean()),
            "std_l2_norm": float(np.linalg.norm(features, axis=1).std()),
            "has_nan": bool(np.isnan(features).any()),
            "has_inf": bool(np.isinf(features).any()),
        }

        # Check if features are approximately normalized
        l2_norms = np.linalg.norm(features, axis=1)
        results["normalized"] = bool(
            np.allclose(l2_norms, 1.0, atol=0.01)
        )

        return results
