"""
Services package for YOLO Editor.
This package contains service classes that handle business logic
and separate concerns from the UI layer.
"""

# Interfaces
from .interfaces import (
    IDatasetService, IImageService, IConfigService, ILogger,
    IEventBus, IProgressReporter, IValidationService, IExportService,
    DatasetStats, ImageInfo, Events
)

# Concrete implementations
from .dataset_service import DatasetService
from .image_service import ImageService
from .config_service import ConfigService, AppConfig, KeymapConfig, UIConfig, EditorConfig
from .logging_service import LoggingService, LogLevel, NullLogger, MemoryLogger
from .event_bus import EventBus, ScopedEventPublisher, EventBusDecorator
from .validation_service import ValidationService, ValidationResult
from .container import (
    ServiceContainer, ServiceContainerBuilder,
    get_container, set_container, get_service, configure_services, inject
)

__all__ = [
    # Interfaces
    'IDatasetService', 'IImageService', 'IConfigService', 'ILogger',
    'IEventBus', 'IProgressReporter', 'IValidationService', 'IExportService',
    'DatasetStats', 'ImageInfo', 'Events',
    
    # Implementations
    'DatasetService', 'ImageService', 'ConfigService', 'LoggingService', 'EventBus', 'ValidationService',
    
    # Configuration classes
    'AppConfig', 'KeymapConfig', 'UIConfig', 'EditorConfig',
    
    # Logging utilities
    'LogLevel', 'NullLogger', 'MemoryLogger',
    
    # Validation utilities
    'ValidationResult',
    
    # Event bus utilities
    'ScopedEventPublisher', 'EventBusDecorator',
    
    # Dependency injection
    'ServiceContainer', 'ServiceContainerBuilder',
    'get_container', 'set_container', 'get_service', 'configure_services', 'inject'
]