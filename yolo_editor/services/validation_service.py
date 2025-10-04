"""
Validation service implementation for YOLO Editor.
Provides comprehensive validation for datasets, images, labels, and configurations.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
import re
from dataclasses import dataclass

from .interfaces import IValidationService, ILogger
from ..core.yolo_io import Box
from ..core.dataset_resolver import DatasetModel


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    issues: List[str]
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class ValidationService(IValidationService):
    """Concrete implementation of validation service."""
    
    # Valid image extensions
    VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.gif'}
    
    # YOLO format constraints
    MIN_COORDINATE = 0.0
    MAX_COORDINATE = 1.0
    MIN_DIMENSION = 1e-6
    MAX_DIMENSION = 1.0
    
    def __init__(self, logger: ILogger):
        self._logger = logger
    
    def validate_image_path(self, path: Path) -> bool:
        """Validate if a path points to a valid image file."""
        try:
            if not path.exists():
                return False
            
            if not path.is_file():
                return False
            
            # Check extension
            if path.suffix.lower() not in self.VALID_IMAGE_EXTENSIONS:
                return False
            
            # Check if file is readable
            try:
                with open(path, 'rb') as f:
                    # Read first few bytes to check if it's a valid image
                    header = f.read(16)
                    if not header:
                        return False
                    
                    # Basic header checks for common formats
                    if self._is_valid_image_header(header, path.suffix.lower()):
                        return True
            except (OSError, IOError):
                return False
            
            return False
            
        except Exception as e:
            self._logger.warning(f"Error validating image path {path}: {e}")
            return False
    
    def _is_valid_image_header(self, header: bytes, extension: str) -> bool:
        """Check if the file header matches the expected format."""
        if extension in {'.jpg', '.jpeg'}:
            return header.startswith(b'\xff\xd8\xff')
        elif extension == '.png':
            return header.startswith(b'\x89PNG\r\n\x1a\n')
        elif extension == '.bmp':
            return header.startswith(b'BM')
        elif extension == '.gif':
            return header.startswith(b'GIF87a') or header.startswith(b'GIF89a')
        elif extension in {'.tif', '.tiff'}:
            return header.startswith(b'II*\x00') or header.startswith(b'MM\x00*')
        elif extension == '.webp':
            return b'WEBP' in header[:12]
        
        # For other formats, assume valid if we got here
        return True
    
    def validate_label_file(self, path: Path) -> Tuple[bool, List[str]]:
        """Validate a YOLO label file and return issues if any."""
        issues = []
        
        try:
            if not path.exists():
                return False, ["Label file does not exist"]
            
            if not path.is_file():
                return False, ["Path is not a file"]
            
            if path.suffix.lower() != '.txt':
                issues.append(f"Unexpected file extension: {path.suffix}")
            
            # Read and validate content
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                return False, ["File is not valid UTF-8 text"]
            except (OSError, IOError) as e:
                return False, [f"Cannot read file: {e}"]
            
            # Validate each line
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue  # Empty lines are OK
                
                line_issues = self._validate_label_line(line, line_num)
                issues.extend(line_issues)
            
            return len(issues) == 0, issues
            
        except Exception as e:
            self._logger.error(f"Error validating label file {path}", exception=e)
            return False, [f"Validation error: {e}"]
    
    def _validate_label_line(self, line: str, line_num: int) -> List[str]:
        """Validate a single line in a YOLO label file."""
        issues = []
        
        # Split into components
        parts = line.split()
        if len(parts) != 5:
            issues.append(f"Line {line_num}: Expected 5 values, got {len(parts)}")
            return issues
        
        try:
            # Parse values
            class_id = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            
            # Validate class ID
            if class_id < 0:
                issues.append(f"Line {line_num}: Class ID must be non-negative, got {class_id}")
            
            # Validate coordinates and dimensions
            coord_issues = self._validate_box_values(cx, cy, w, h, line_num)
            issues.extend(coord_issues)
            
        except ValueError as e:
            issues.append(f"Line {line_num}: Invalid number format - {e}")
        
        return issues
    
    def _validate_box_values(self, cx: float, cy: float, w: float, h: float, line_num: int) -> List[str]:
        """Validate bounding box coordinate values."""
        issues = []
        
        # Check coordinate ranges
        if not (self.MIN_COORDINATE <= cx <= self.MAX_COORDINATE):
            issues.append(f"Line {line_num}: Center X {cx} out of range [0, 1]")
        
        if not (self.MIN_COORDINATE <= cy <= self.MAX_COORDINATE):
            issues.append(f"Line {line_num}: Center Y {cy} out of range [0, 1]")
        
        # Check dimension ranges
        if not (self.MIN_DIMENSION <= w <= self.MAX_DIMENSION):
            issues.append(f"Line {line_num}: Width {w} out of range [{self.MIN_DIMENSION}, 1]")
        
        if not (self.MIN_DIMENSION <= h <= self.MAX_DIMENSION):
            issues.append(f"Line {line_num}: Height {h} out of range [{self.MIN_DIMENSION}, 1]")
        
        # Check if box extends beyond image boundaries
        half_w = w / 2
        half_h = h / 2
        
        if cx - half_w < 0:
            issues.append(f"Line {line_num}: Box extends beyond left edge")
        
        if cx + half_w > 1:
            issues.append(f"Line {line_num}: Box extends beyond right edge")
        
        if cy - half_h < 0:
            issues.append(f"Line {line_num}: Box extends beyond top edge")
        
        if cy + half_h > 1:
            issues.append(f"Line {line_num}: Box extends beyond bottom edge")
        
        return issues
    
    def validate_box(self, box: Box, image_size: Optional[Tuple[int, int]] = None) -> Tuple[bool, List[str]]:
        """Validate a bounding box."""
        issues = []
        
        # Validate class ID
        if box.cls < 0:
            issues.append(f"Class ID must be non-negative, got {box.cls}")
        
        # Validate coordinates and dimensions
        coord_issues = self._validate_box_values(box.cx, box.cy, box.w, box.h, 0)
        # Remove line number references for box validation
        coord_issues = [issue.replace("Line 0: ", "") for issue in coord_issues]
        issues.extend(coord_issues)
        
        # Additional validation with image size
        if image_size:
            img_w, img_h = image_size
            if img_w <= 0 or img_h <= 0:
                issues.append(f"Invalid image size: {image_size}")
            else:
                # Check minimum pixel size
                min_pixel_w = 2.0 / img_w
                min_pixel_h = 2.0 / img_h
                
                if box.w < min_pixel_w:
                    issues.append(f"Box width too small: {box.w} (min: {min_pixel_w:.6f})")
                
                if box.h < min_pixel_h:
                    issues.append(f"Box height too small: {box.h} (min: {min_pixel_h:.6f})")
        
        return len(issues) == 0, issues
    
    def validate_dataset(self, dataset: DatasetModel) -> ValidationResult:
        """Validate an entire dataset."""
        issues = []
        warnings = []
        
        # Check basic dataset structure
        if not dataset.splits:
            issues.append("Dataset has no splits")
            return ValidationResult(False, issues, warnings)
        
        # Validate each split
        for split_name, split_info in dataset.splits.items():
            split_issues, split_warnings = self._validate_split(split_name, split_info)
            issues.extend(split_issues)
            warnings.extend(split_warnings)
        
        # Validate class names
        if dataset.names:
            name_issues = self._validate_class_names(dataset.names)
            issues.extend(name_issues)
        else:
            warnings.append("No class names defined")
        
        return ValidationResult(len(issues) == 0, issues, warnings)
    
    def _validate_split(self, split_name: str, split_info: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate a single dataset split."""
        issues = []
        warnings = []
        
        images = split_info.get("images", [])
        images_dir = split_info.get("images_dir")
        labels_dir = split_info.get("labels_dir")
        
        if not images:
            issues.append(f"Split '{split_name}' has no images")
            return issues, warnings
        
        if not images_dir or not images_dir.exists():
            issues.append(f"Split '{split_name}' images directory missing: {images_dir}")
        
        if labels_dir and not labels_dir.exists():
            warnings.append(f"Split '{split_name}' labels directory missing: {labels_dir}")
        
        # Sample validation of images and labels
        sample_size = min(10, len(images))
        for img_path in images[:sample_size]:
            if not self.validate_image_path(img_path):
                issues.append(f"Invalid image in '{split_name}': {img_path.name}")
        
        return issues, warnings
    
    def _validate_class_names(self, names: List[str]) -> List[str]:
        """Validate class names."""
        issues = []
        
        for i, name in enumerate(names):
            if not name or not isinstance(name, str):
                issues.append(f"Invalid class name at index {i}: {repr(name)}")
            elif not name.strip():
                issues.append(f"Empty class name at index {i}")
            elif len(name) > 100:
                issues.append(f"Class name too long at index {i}: {len(name)} characters")
        
        # Check for duplicates
        seen = set()
        for i, name in enumerate(names):
            if isinstance(name, str) and name in seen:
                issues.append(f"Duplicate class name at index {i}: '{name}'")
            seen.add(name)
        
        return issues
    
    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate application configuration."""
        issues = []
        warnings = []
        
        # Validate keymap section
        if 'keymap' in config:
            keymap_issues = self._validate_keymap(config['keymap'])
            issues.extend(keymap_issues)
        
        # Validate UI section
        if 'ui' in config:
            ui_issues, ui_warnings = self._validate_ui_config(config['ui'])
            issues.extend(ui_issues)
            warnings.extend(ui_warnings)
        
        # Validate editor section
        if 'editor' in config:
            editor_issues = self._validate_editor_config(config['editor'])
            issues.extend(editor_issues)
        
        return ValidationResult(len(issues) == 0, issues, warnings)
    
    def _validate_keymap(self, keymap: Dict[str, Any]) -> List[str]:
        """Validate keymap configuration."""
        issues = []
        
        required_keys = {
            'next_image', 'prev_image', 'save', 'add_box', 
            'delete_box', 'change_class', 'fit', 'zoom_100'
        }
        
        for key in required_keys:
            if key not in keymap:
                issues.append(f"Missing keymap entry: {key}")
            elif not isinstance(keymap[key], str):
                issues.append(f"Keymap entry '{key}' must be a string")
            elif not keymap[key].strip():
                issues.append(f"Keymap entry '{key}' cannot be empty")
        
        return issues
    
    def _validate_ui_config(self, ui_config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate UI configuration."""
        issues = []
        warnings = []
        
        # Validate window dimensions
        if 'window_width' in ui_config:
            width = ui_config['window_width']
            if not isinstance(width, int) or width < 400:
                issues.append(f"Invalid window width: {width}")
        
        if 'window_height' in ui_config:
            height = ui_config['window_height']
            if not isinstance(height, int) or height < 300:
                issues.append(f"Invalid window height: {height}")
        
        # Validate theme
        if 'theme' in ui_config:
            theme = ui_config['theme']
            valid_themes = {'light', 'dark', 'auto'}
            if theme not in valid_themes:
                warnings.append(f"Unknown theme: {theme}")
        
        # Validate recent files
        if 'recent_files' in ui_config:
            recent = ui_config['recent_files']
            if not isinstance(recent, list):
                issues.append("Recent files must be a list")
            elif len(recent) > 20:
                warnings.append(f"Too many recent files: {len(recent)}")
        
        return issues, warnings
    
    def _validate_editor_config(self, editor_config: Dict[str, Any]) -> List[str]:
        """Validate editor configuration."""
        issues = []
        
        # Validate boolean settings
        bool_settings = {'auto_save', 'show_class_names'}
        for setting in bool_settings:
            if setting in editor_config and not isinstance(editor_config[setting], bool):
                issues.append(f"Setting '{setting}' must be boolean")
        
        # Validate numeric settings
        if 'box_line_width' in editor_config:
            width = editor_config['box_line_width']
            if not isinstance(width, int) or width < 1 or width > 10:
                issues.append(f"Invalid box line width: {width}")
        
        # Validate color settings
        color_settings = {'selected_box_color', 'default_box_color', 'background_color'}
        for setting in color_settings:
            if setting in editor_config:
                color = editor_config[setting]
                if not self._is_valid_color(color):
                    issues.append(f"Invalid color for '{setting}': {color}")
        
        return issues
    
    def _is_valid_color(self, color: str) -> bool:
        """Validate a color string (hex format)."""
        if not isinstance(color, str):
            return False
        
        # Check hex color format
        hex_pattern = re.compile(r'^#[0-9a-fA-F]{6}$')
        return bool(hex_pattern.match(color))
    
    def validate_export_options(self, options: Dict[str, Any]) -> ValidationResult:
        """Validate dataset export options."""
        issues = []
        warnings = []
        
        # Validate output path
        if 'output_path' not in options:
            issues.append("Output path is required")
        else:
            output_path = Path(options['output_path'])
            if output_path.exists() and not output_path.is_dir():
                issues.append("Output path must be a directory")
        
        # Validate class mapping
        if 'class_mapping' in options:
            mapping = options['class_mapping']
            if not isinstance(mapping, dict):
                issues.append("Class mapping must be a dictionary")
            else:
                for src_cls, tgt_cls in mapping.items():
                    if not isinstance(src_cls, int) or src_cls < 0:
                        issues.append(f"Invalid source class ID: {src_cls}")
                    if not isinstance(tgt_cls, int) or tgt_cls < 0:
                        issues.append(f"Invalid target class ID: {tgt_cls}")
        
        # Validate other options
        if 'copy_images' in options and not isinstance(options['copy_images'], bool):
            issues.append("copy_images option must be boolean")
        
        if 'dry_run' in options and not isinstance(options['dry_run'], bool):
            issues.append("dry_run option must be boolean")
        
        return ValidationResult(len(issues) == 0, issues, warnings)