"""
Dataset service implementation for YOLO Editor.
Handles dataset loading, validation, and statistics calculation.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict

from .interfaces import IDatasetService, IImageService, ILogger, DatasetStats
from ..core.dataset_resolver import DatasetModel, resolve_dataset
from ..core.yolo_io import Box, read_yolo_txt, labels_for_image


class DatasetService(IDatasetService):
    """Concrete implementation of dataset service."""
    
    def __init__(self, image_service: IImageService, logger: ILogger):
        self._image_service = image_service
        self._logger = logger
        self._cache: Dict[str, DatasetModel] = {}
    
    def load_dataset(self, path: Path) -> DatasetModel:
        """Load a dataset from a path (directory or YAML file)."""
        try:
            cache_key = str(path.resolve())
            if cache_key in self._cache:
                self._logger.debug(f"Loading dataset from cache: {path}")
                return self._cache[cache_key]
            
            self._logger.info(f"Loading dataset from: {path}")
            dataset = resolve_dataset(path)
            
            if not dataset.splits:
                raise ValueError(f"No valid splits found in dataset at {path}")
            
            # Validate the dataset
            issues = self.validate_dataset(dataset)
            if issues:
                self._logger.warning(f"Dataset validation issues: {'; '.join(issues)}")
            
            self._cache[cache_key] = dataset
            self._logger.info(f"Successfully loaded dataset with {len(dataset.splits)} splits")
            return dataset
            
        except Exception as e:
            self._logger.error(f"Failed to load dataset from {path}", exception=e)
            raise
    
    def get_dataset_stats(self, dataset: DatasetModel, split: str) -> DatasetStats:
        """Calculate statistics for a dataset split."""
        if split not in dataset.splits:
            raise ValueError(f"Split '{split}' not found in dataset")
        
        split_info = dataset.splits[split]
        images = split_info.get("images", [])
        labels_dir = split_info.get("labels_dir")
        images_dir = split_info.get("images_dir")
        
        self._logger.debug(f"Calculating stats for split '{split}' with {len(images)} images")
        
        per_class_images = defaultdict(int)
        per_class_boxes = defaultdict(int)
        folder_stats: Dict[str, Dict[str, Any]] = {}
        images_without_labels = []
        
        for img_path in images:
            try:
                # Load labels for this image
                boxes = self._image_service.load_labels(img_path, labels_dir, images_dir)
                
                if not boxes:
                    images_without_labels.append(img_path)
                
                # Get image size for validation
                img_size = self._image_service.get_image_size(img_path)
                if img_size:
                    boxes, _ = self._image_service.sanitize_boxes(boxes, img_size)
                
                # Calculate folder stats
                rel_folder = "."
                if images_dir:
                    try:
                        rel = img_path.relative_to(images_dir)
                        rel_folder = rel.parent.as_posix() or "."
                    except ValueError:
                        rel_folder = img_path.parent.as_posix()
                
                folder_entry = folder_stats.setdefault(rel_folder, {
                    "images": 0, 
                    "per_class": defaultdict(int)
                })
                folder_entry["images"] += 1
                
                # Count classes
                seen_classes = set()
                for box in boxes:
                    per_class_boxes[box.cls] += 1
                    seen_classes.add(box.cls)
                    folder_entry["per_class"][box.cls] += 1
                
                for cls_id in seen_classes:
                    per_class_images[cls_id] += 1
                    
            except Exception as e:
                self._logger.warning(f"Error processing image {img_path}: {e}")
                continue
        
        # Convert defaultdicts to regular dicts
        folder_payload = {}
        for folder, info in folder_stats.items():
            folder_payload[folder] = {
                "images": info["images"],
                "per_class": dict(info["per_class"]),
            }
        
        stats = DatasetStats(
            total_images=len(images),
            total_boxes=sum(per_class_boxes.values()),
            images_with_labels=len(images) - len(images_without_labels),
            images_without_labels=len(images_without_labels),
            per_class_images=dict(per_class_images),
            per_class_boxes=dict(per_class_boxes),
            folder_stats=folder_payload
        )
        
        self._logger.debug(f"Stats calculated: {stats.total_images} images, {stats.total_boxes} boxes")
        return stats
    
    def validate_dataset(self, dataset: DatasetModel) -> List[str]:
        """Validate a dataset and return any issues found."""
        issues = []
        
        if not dataset.splits:
            issues.append("No splits found in dataset")
            return issues
        
        for split_name, split_info in dataset.splits.items():
            images = split_info.get("images", [])
            labels_dir = split_info.get("labels_dir")
            images_dir = split_info.get("images_dir")
            
            if not images:
                issues.append(f"Split '{split_name}' has no images")
                continue
            
            if not images_dir or not images_dir.exists():
                issues.append(f"Split '{split_name}' images directory does not exist: {images_dir}")
            
            if labels_dir and not labels_dir.exists():
                issues.append(f"Split '{split_name}' labels directory does not exist: {labels_dir}")
            
            # Sample a few images to check validity
            sample_size = min(10, len(images))
            for img_path in images[:sample_size]:
                if not self._image_service.load_image(img_path):
                    issues.append(f"Cannot load image: {img_path}")
        
        # Check class names consistency
        if dataset.names:
            for i, name in enumerate(dataset.names):
                if not name or not isinstance(name, str):
                    issues.append(f"Invalid class name at index {i}: {name}")
        
        return issues
    
    def clear_cache(self) -> None:
        """Clear the dataset cache."""
        self._cache.clear()
        self._logger.debug("Dataset cache cleared")
    
    def get_cached_datasets(self) -> List[str]:
        """Get list of cached dataset paths."""
        return list(self._cache.keys())