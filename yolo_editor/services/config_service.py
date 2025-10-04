"""
Configuration service implementation for YOLO Editor.
Handles application settings and configuration management.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
import json
import yaml
from dataclasses import dataclass, asdict

from .interfaces import IConfigService, ILogger


@dataclass
class KeymapConfig:
    """Keyboard shortcut configuration."""
    next_image: str = "N"
    prev_image: str = "P"
    save: str = "S"
    add_box: str = "A"
    delete_box: str = "Delete"
    change_class: str = "C"
    fit: str = "F"
    zoom_100: str = "1"


@dataclass
class UIConfig:
    """UI-related configuration."""
    theme: str = "dark"
    window_width: int = 1380
    window_height: int = 860
    splitter_sizes: list = None
    recent_files: list = None
    
    def __post_init__(self):
        if self.splitter_sizes is None:
            self.splitter_sizes = [300, 800, 280]
        if self.recent_files is None:
            self.recent_files = []


@dataclass
class EditorConfig:
    """Editor-specific configuration."""
    auto_save: bool = False
    show_class_names: bool = True
    box_line_width: int = 2
    selected_box_color: str = "#ff0000"
    default_box_color: str = "#00ff00"
    background_color: str = "#2b2b2b"


@dataclass
class AppConfig:
    """Main application configuration."""
    keymap: KeymapConfig = None
    ui: UIConfig = None
    editor: EditorConfig = None
    
    def __post_init__(self):
        if self.keymap is None:
            self.keymap = KeymapConfig()
        if self.ui is None:
            self.ui = UIConfig()
        if self.editor is None:
            self.editor = EditorConfig()


class ConfigService(IConfigService):
    """Concrete implementation of configuration service."""
    
    def __init__(self, logger: ILogger, config_dir: Optional[Path] = None):
        self._logger = logger
        self._config_dir = config_dir or self._get_default_config_dir()
        self._config_file = self._config_dir / "config.json"
        self._config: AppConfig = AppConfig()
        self._ensure_config_dir()
        self._load_config_from_file()
    
    def _get_default_config_dir(self) -> Path:
        """Get the default configuration directory."""
        import os
        if os.name == 'nt':  # Windows
            config_dir = Path.home() / "AppData" / "Local" / "YOLOEditor"
        else:  # Unix-like
            config_dir = Path.home() / ".config" / "yolo-editor"
        return config_dir
    
    def _ensure_config_dir(self) -> None:
        """Ensure the configuration directory exists."""
        try:
            self._config_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self._logger.error(f"Failed to create config directory: {self._config_dir}", exception=e)
    
    def _load_config_from_file(self) -> None:
        """Load configuration from file."""
        if not self._config_file.exists():
            self._logger.info("No config file found, using defaults")
            return
        
        try:
            with open(self._config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert dict to config objects
            self._config = self._dict_to_config(data)
            self._logger.info(f"Loaded configuration from: {self._config_file}")
            
        except Exception as e:
            self._logger.error(f"Failed to load config from {self._config_file}", exception=e)
            self._config = AppConfig()  # Use defaults
    
    def _dict_to_config(self, data: Dict[str, Any]) -> AppConfig:
        """Convert dictionary to AppConfig object."""
        config = AppConfig()
        
        if 'keymap' in data:
            config.keymap = KeymapConfig(**data['keymap'])
        
        if 'ui' in data:
            config.ui = UIConfig(**data['ui'])
        
        if 'editor' in data:
            config.editor = EditorConfig(**data['editor'])
        
        return config
    
    def _config_to_dict(self, config: AppConfig) -> Dict[str, Any]:
        """Convert AppConfig object to dictionary."""
        return {
            'keymap': asdict(config.keymap),
            'ui': asdict(config.ui),
            'editor': asdict(config.editor)
        }
    
    def load_config(self) -> Dict[str, Any]:
        """Load application configuration."""
        return self._config_to_dict(self._config)
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save application configuration."""
        try:
            # Update internal config
            self._config = self._dict_to_config(config)
            
            # Save to file
            with open(self._config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            self._logger.info(f"Saved configuration to: {self._config_file}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to save config to {self._config_file}", exception=e)
            return False
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a specific setting value using dot notation."""
        try:
            parts = key.split('.')
            value = self._config
            
            for part in parts:
                if hasattr(value, part):
                    value = getattr(value, part)
                else:
                    return default
            
            return value
            
        except Exception as e:
            self._logger.warning(f"Failed to get setting '{key}': {e}")
            return default
    
    def set_setting(self, key: str, value: Any) -> None:
        """Set a specific setting value using dot notation."""
        try:
            parts = key.split('.')
            if len(parts) < 2:
                self._logger.warning(f"Invalid setting key format: {key}")
                return
            
            # Navigate to the parent object
            obj = self._config
            for part in parts[:-1]:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    self._logger.warning(f"Setting path not found: {key}")
                    return
            
            # Set the final value
            final_key = parts[-1]
            if hasattr(obj, final_key):
                setattr(obj, final_key, value)
                self._logger.debug(f"Set setting '{key}' = {value}")
            else:
                self._logger.warning(f"Setting key not found: {key}")
                
        except Exception as e:
            self._logger.error(f"Failed to set setting '{key}': {e}")
    
    def get_keymap(self) -> KeymapConfig:
        """Get keyboard mapping configuration."""
        return self._config.keymap
    
    def get_ui_config(self) -> UIConfig:
        """Get UI configuration."""
        return self._config.ui
    
    def get_editor_config(self) -> EditorConfig:
        """Get editor configuration."""
        return self._config.editor
    
    def add_recent_file(self, file_path: str) -> None:
        """Add a file to the recent files list."""
        recent = self._config.ui.recent_files
        
        # Remove if already exists
        if file_path in recent:
            recent.remove(file_path)
        
        # Add to beginning
        recent.insert(0, file_path)
        
        # Limit to 10 recent files
        self._config.ui.recent_files = recent[:10]
        
        # Auto-save
        self.save_config(self.load_config())
    
    def get_recent_files(self) -> list:
        """Get list of recent files."""
        return self._config.ui.recent_files.copy()
    
    def export_config(self, export_path: Path) -> bool:
        """Export configuration to a file."""
        try:
            config_dict = self.load_config()
            
            if export_path.suffix.lower() == '.yaml':
                with open(export_path, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(config_dict, f, default_flow_style=False)
            else:
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            self._logger.info(f"Exported configuration to: {export_path}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to export config to {export_path}", exception=e)
            return False
    
    def import_config(self, import_path: Path) -> bool:
        """Import configuration from a file."""
        try:
            if not import_path.exists():
                self._logger.error(f"Config file not found: {import_path}")
                return False
            
            with open(import_path, 'r', encoding='utf-8') as f:
                if import_path.suffix.lower() == '.yaml':
                    config_dict = yaml.safe_load(f)
                else:
                    config_dict = json.load(f)
            
            success = self.save_config(config_dict)
            if success:
                self._logger.info(f"Imported configuration from: {import_path}")
            
            return success
            
        except Exception as e:
            self._logger.error(f"Failed to import config from {import_path}", exception=e)
            return False