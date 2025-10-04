"""
Abstract interfaces for YOLO Editor services.
These interfaces define contracts for different service components,
enabling better testability and maintainability.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any, Callable
from pathlib import Path
from dataclasses import dataclass

from ..core.dataset_resolver import DatasetModel
from ..core.yolo_io import Box


@dataclass
class ImageInfo:
    """Information about an image file."""
    path: Path
    size: Optional[Tuple[int, int]] = None
    boxes: Optional[List[Box]] = None


@dataclass
class DatasetStats:
    """Statistics for a dataset."""
    total_images: int
    total_boxes: int
    images_with_labels: int
    images_without_labels: int
    per_class_images: Dict[int, int]
    per_class_boxes: Dict[int, int]
    folder_stats: Dict[str, Dict[str, Any]]


class IDatasetService(ABC):
    """Interface for dataset management operations."""
    
    @abstractmethod
    def load_dataset(self, path: Path) -> DatasetModel:
        """Load a dataset from a path (directory or YAML file)."""
        pass
    
    @abstractmethod
    def get_dataset_stats(self, dataset: DatasetModel, split: str) -> DatasetStats:
        """Calculate statistics for a dataset split."""
        pass
    
    @abstractmethod
    def validate_dataset(self, dataset: DatasetModel) -> List[str]:
        """Validate a dataset and return any issues found."""
        pass


class IImageService(ABC):
    """Interface for image and label operations."""
    
    @abstractmethod
    def load_image(self, path: Path) -> Optional[Any]:
        """Load an image from file."""
        pass
    
    @abstractmethod
    def get_image_size(self, path: Path) -> Optional[Tuple[int, int]]:
        """Get image dimensions."""
        pass
    
    @abstractmethod
    def load_labels(self, image_path: Path, labels_dir: Optional[Path], 
                   images_dir: Optional[Path]) -> List[Box]:
        """Load YOLO labels for an image."""
        pass
    
    @abstractmethod
    def save_labels(self, image_path: Path, labels_dir: Optional[Path], 
                   images_dir: Optional[Path], boxes: List[Box]) -> bool:
        """Save YOLO labels for an image."""
        pass
    
    @abstractmethod
    def sanitize_boxes(self, boxes: List[Box], image_size: Tuple[int, int]) -> Tuple[List[Box], bool]:
        """Sanitize boxes to ensure they're within valid bounds."""
        pass


class IConfigService(ABC):
    """Interface for configuration management."""
    
    @abstractmethod
    def load_config(self) -> Dict[str, Any]:
        """Load application configuration."""
        pass
    
    @abstractmethod
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save application configuration."""
        pass
    
    @abstractmethod
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a specific setting value."""
        pass
    
    @abstractmethod
    def set_setting(self, key: str, value: Any) -> None:
        """Set a specific setting value."""
        pass


class IProgressReporter(ABC):
    """Interface for progress reporting."""
    
    @abstractmethod
    def start_operation(self, title: str, total_steps: int = 0) -> None:
        """Start a new operation with progress tracking."""
        pass
    
    @abstractmethod
    def update_progress(self, current: int, message: str = "") -> None:
        """Update progress for the current operation."""
        pass
    
    @abstractmethod
    def finish_operation(self, success: bool = True, message: str = "") -> None:
        """Finish the current operation."""
        pass
    
    @abstractmethod
    def is_cancelled(self) -> bool:
        """Check if the current operation was cancelled."""
        pass


class IEventBus(ABC):
    """Interface for event-driven communication between components."""
    
    @abstractmethod
    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to an event type."""
        pass
    
    @abstractmethod
    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe from an event type."""
        pass
    
    @abstractmethod
    def publish(self, event_type: str, data: Any = None) -> None:
        """Publish an event."""
        pass


class ILogger(ABC):
    """Interface for logging operations."""
    
    @abstractmethod
    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message."""
        pass
    
    @abstractmethod
    def info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        pass
    
    @abstractmethod
    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        pass
    
    @abstractmethod
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log an error message."""
        pass


class IValidationService(ABC):
    """Interface for data validation operations."""
    
    @abstractmethod
    def validate_image_path(self, path: Path) -> bool:
        """Validate if a path points to a valid image file."""
        pass
    
    @abstractmethod
    def validate_label_file(self, path: Path) -> Tuple[bool, List[str]]:
        """Validate a YOLO label file and return issues if any."""
        pass
    
    @abstractmethod
    def validate_box(self, box: Box, image_size: Optional[Tuple[int, int]] = None) -> Tuple[bool, List[str]]:
        """Validate a bounding box."""
        pass


class IExportService(ABC):
    """Interface for dataset export operations."""
    
    @abstractmethod
    def export_dataset(self, source_dataset: DatasetModel, output_path: Path, 
                      class_mapping: Dict[int, int], options: Dict[str, Any]) -> bool:
        """Export a dataset with class remapping."""
        pass
    
    @abstractmethod
    def create_manifest(self, export_path: Path, metadata: Dict[str, Any]) -> Path:
        """Create an export manifest file."""
        pass


# Event types for the event bus
class Events:
    """Standard event types used throughout the application."""
    
    # Dataset events
    DATASET_LOADED = "dataset.loaded"
    DATASET_CHANGED = "dataset.changed"
    DATASET_STATS_UPDATED = "dataset.stats_updated"
    
    # Image events
    IMAGE_OPENED = "image.opened"
    IMAGE_CHANGED = "image.changed"
    LABELS_CHANGED = "labels.changed"
    LABELS_SAVED = "labels.saved"
    
    # UI events
    SELECTION_CHANGED = "ui.selection_changed"
    CLASS_CHANGED = "ui.class_changed"
    ZOOM_CHANGED = "ui.zoom_changed"
    
    # Application events
    CONFIG_CHANGED = "app.config_changed"
    ERROR_OCCURRED = "app.error_occurred"
    OPERATION_STARTED = "app.operation_started"
    OPERATION_FINISHED = "app.operation_finished"