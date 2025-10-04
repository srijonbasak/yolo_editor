"""
Image service implementation for YOLO Editor.
Handles image loading, label operations, and box validation.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Any, Dict
from pathlib import Path
import numpy as np

from .interfaces import IImageService, ILogger
from ..core.yolo_io import Box, read_yolo_txt, write_yolo_txt, labels_for_image, imread_unicode


class ImageService(IImageService):
    """Concrete implementation of image service."""
    
    # Constants for box validation
    MIN_BOX_NORM = 1e-4
    MIN_BOX_PIXELS = 2
    
    def __init__(self, logger: ILogger):
        self._logger = logger
        self._image_cache: Dict[str, Any] = {}
        self._size_cache: Dict[str, Tuple[int, int]] = {}
        self._label_cache: Dict[str, List[Box]] = {}
    
    def load_image(self, path: Path) -> Optional[Any]:
        """Load an image from file."""
        try:
            cache_key = str(path.resolve())
            if cache_key in self._image_cache:
                return self._image_cache[cache_key]
            
            self._logger.debug(f"Loading image: {path}")
            image = imread_unicode(path)
            
            if image is None:
                self._logger.warning(f"Failed to load image: {path}")
                return None
            
            # Cache the image (be careful with memory usage)
            if len(self._image_cache) < 10:  # Limit cache size
                self._image_cache[cache_key] = image.copy()
            
            return image
            
        except Exception as e:
            self._logger.error(f"Error loading image {path}", exception=e)
            return None
    
    def get_image_size(self, path: Path) -> Optional[Tuple[int, int]]:
        """Get image dimensions."""
        try:
            cache_key = str(path.resolve())
            if cache_key in self._size_cache:
                return self._size_cache[cache_key]
            
            # Try to get size from cached image first
            if cache_key in self._image_cache:
                image = self._image_cache[cache_key]
                size = (image.shape[1], image.shape[0])
                self._size_cache[cache_key] = size
                return size
            
            # Load image to get size
            image = self.load_image(path)
            if image is None:
                return None
            
            size = (image.shape[1], image.shape[0])
            self._size_cache[cache_key] = size
            return size
            
        except Exception as e:
            self._logger.error(f"Error getting image size for {path}", exception=e)
            return None
    
    def load_labels(self, image_path: Path, labels_dir: Optional[Path], 
                   images_dir: Optional[Path]) -> List[Box]:
        """Load YOLO labels for an image."""
        try:
            cache_key = f"{image_path}:{labels_dir}:{images_dir}"
            if cache_key in self._label_cache:
                return self._label_cache[cache_key].copy()
            
            label_path = labels_for_image(image_path, labels_dir, images_dir)
            boxes = read_yolo_txt(label_path)
            
            # Sanitize boxes if we have image size
            image_size = self.get_image_size(image_path)
            if image_size:
                boxes, changed = self.sanitize_boxes(boxes, image_size)
                if changed:
                    self._logger.debug(f"Sanitized boxes for image: {image_path}")
            
            # Cache the result
            self._label_cache[cache_key] = boxes.copy()
            return boxes
            
        except Exception as e:
            self._logger.error(f"Error loading labels for {image_path}", exception=e)
            return []
    
    def save_labels(self, image_path: Path, labels_dir: Optional[Path], 
                   images_dir: Optional[Path], boxes: List[Box]) -> bool:
        """Save YOLO labels for an image."""
        try:
            label_path = labels_for_image(image_path, labels_dir, images_dir)
            
            # Sanitize boxes before saving
            image_size = self.get_image_size(image_path)
            if image_size:
                boxes, changed = self.sanitize_boxes(boxes, image_size)
                if changed:
                    self._logger.debug(f"Sanitized boxes before saving: {image_path}")
            
            write_yolo_txt(label_path, boxes)
            
            # Update cache
            cache_key = f"{image_path}:{labels_dir}:{images_dir}"
            self._label_cache[cache_key] = boxes.copy()
            
            self._logger.debug(f"Saved {len(boxes)} boxes to: {label_path}")
            return True
            
        except Exception as e:
            self._logger.error(f"Error saving labels for {image_path}", exception=e)
            return False
    
    def sanitize_boxes(self, boxes: List[Box], image_size: Tuple[int, int]) -> Tuple[List[Box], bool]:
        """Sanitize boxes to ensure they're within valid bounds."""
        if not boxes or image_size[0] <= 0 or image_size[1] <= 0:
            return list(boxes), False
        
        img_w, img_h = image_size
        min_w = max(self.MIN_BOX_PIXELS / img_w, self.MIN_BOX_NORM)
        min_h = max(self.MIN_BOX_PIXELS / img_h, self.MIN_BOX_NORM)
        
        sanitized: List[Box] = []
        changed = False
        
        for box in boxes:
            # Clamp dimensions
            new_w = max(min(box.w, 1.0), min_w)
            new_h = max(min(box.h, 1.0), min_h)
            
            # Calculate valid center bounds
            half_w = min(new_w / 2, 0.5)
            half_h = min(new_h / 2, 0.5)
            
            # Clamp center coordinates
            if new_w >= 1.0:
                cx = 0.5
            else:
                cx = min(max(box.cx, half_w), 1.0 - half_w)
            
            if new_h >= 1.0:
                cy = 0.5
            else:
                cy = min(max(box.cy, half_h), 1.0 - half_h)
            
            # Check if anything changed
            if (abs(new_w - box.w) > 1e-6 or abs(new_h - box.h) > 1e-6 or 
                abs(cx - box.cx) > 1e-6 or abs(cy - box.cy) > 1e-6):
                changed = True
            
            sanitized.append(Box(box.cls, cx, cy, new_w, new_h))
        
        return sanitized, changed
    
    def validate_image_path(self, path: Path) -> bool:
        """Check if a path points to a valid image file."""
        if not path.exists() or not path.is_file():
            return False
        
        # Check file extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
        if path.suffix.lower() not in valid_extensions:
            return False
        
        # Try to load the image
        return self.load_image(path) is not None
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self._image_cache.clear()
        self._size_cache.clear()
        self._label_cache.clear()
        self._logger.debug("Image service caches cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "images": len(self._image_cache),
            "sizes": len(self._size_cache),
            "labels": len(self._label_cache)
        }
    
    def preload_images(self, paths: List[Path], max_count: int = 5) -> None:
        """Preload a set of images into cache."""
        for i, path in enumerate(paths[:max_count]):
            if len(self._image_cache) >= 10:  # Respect cache limit
                break
            try:
                self.load_image(path)
                self._logger.debug(f"Preloaded image {i+1}/{min(max_count, len(paths))}: {path}")
            except Exception as e:
                self._logger.warning(f"Failed to preload image {path}: {e}")