# YOLO Editor - Refactored Architecture

## Overview

This document describes the refactored architecture of YOLO Editor, which has been redesigned to improve maintainability, testability, and feature extensibility.

## Architecture Principles

### 1. Separation of Concerns
- **UI Layer**: Handles user interface and user interactions
- **Service Layer**: Contains business logic and data operations
- **Core Layer**: Contains domain models and utilities

### 2. Dependency Injection
- Services are injected through constructor parameters
- Enables easy testing with mock implementations
- Centralized service configuration and lifecycle management

### 3. Event-Driven Architecture
- Components communicate through events rather than direct coupling
- Enables loose coupling and easier feature additions
- Centralized event bus for cross-component communication

### 4. Interface-Based Design
- All services implement well-defined interfaces
- Enables easy swapping of implementations
- Improves testability and maintainability

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                        UI Layer                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   MainWindow    │  │   ImageView     │  │ MergeDesigner│ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                     Service Layer                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ DatasetService  │  │  ImageService   │  │ConfigService │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ LoggingService  │  │   EventBus      │  │   Container  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                      Core Layer                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │     Models      │  │   YOLO I/O      │  │  Validators  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### Service Layer

#### 1. DatasetService (`IDatasetService`)
**Responsibilities:**
- Loading and validating YOLO datasets
- Calculating dataset statistics
- Caching dataset information
- Resolving dataset paths and structures

**Key Methods:**
- `load_dataset(path: Path) -> DatasetModel`
- `get_dataset_stats(dataset: DatasetModel, split: str) -> DatasetStats`
- `validate_dataset(dataset: DatasetModel) -> List[str]`

#### 2. ImageService (`IImageService`)
**Responsibilities:**
- Loading and caching images
- Managing YOLO label files
- Validating and sanitizing bounding boxes
- Image size calculation and caching

**Key Methods:**
- `load_image(path: Path) -> Optional[Any]`
- `load_labels(image_path: Path, ...) -> List[Box]`
- `save_labels(image_path: Path, ..., boxes: List[Box]) -> bool`
- `sanitize_boxes(boxes: List[Box], image_size: Tuple[int, int]) -> Tuple[List[Box], bool]`

#### 3. ConfigService (`IConfigService`)
**Responsibilities:**
- Managing application configuration
- Persisting user preferences
- Handling keyboard shortcuts and UI settings
- Configuration import/export

**Key Methods:**
- `load_config() -> Dict[str, Any]`
- `save_config(config: Dict[str, Any]) -> bool`
- `get_setting(key: str, default: Any) -> Any`
- `set_setting(key: str, value: Any) -> None`

#### 4. LoggingService (`ILogger`)
**Responsibilities:**
- Structured logging with multiple levels
- File and console output
- Performance and user action tracking
- Error reporting with context

**Key Methods:**
- `debug(message: str, **kwargs) -> None`
- `info(message: str, **kwargs) -> None`
- `warning(message: str, **kwargs) -> None`
- `error(message: str, exception: Optional[Exception], **kwargs) -> None`

#### 5. EventBus (`IEventBus`)
**Responsibilities:**
- Decoupled component communication
- Event subscription and publishing
- Weak reference management to prevent memory leaks
- Event history and debugging

**Key Methods:**
- `subscribe(event_type: str, handler: Callable) -> None`
- `unsubscribe(event_type: str, handler: Callable) -> None`
- `publish(event_type: str, data: Any) -> None`

### Dependency Injection Container

The `ServiceContainer` manages service instances and their dependencies:

```python
# Service registration
container = ServiceContainerBuilder()
    .configure_default_services()
    .build()

# Service resolution with automatic dependency injection
dataset_service = container.get(IDatasetService)
```

**Features:**
- Automatic dependency resolution
- Singleton and transient service lifetimes
- Factory-based service creation
- Circular dependency detection

## Event System

### Standard Events

The application uses a standardized set of events defined in `Events` class:

```python
# Dataset events
Events.DATASET_LOADED = "dataset.loaded"
Events.DATASET_CHANGED = "dataset.changed"
Events.DATASET_STATS_UPDATED = "dataset.stats_updated"

# Image events
Events.IMAGE_OPENED = "image.opened"
Events.LABELS_CHANGED = "labels.changed"
Events.LABELS_SAVED = "labels.saved"

# UI events
Events.SELECTION_CHANGED = "ui.selection_changed"
Events.CLASS_CHANGED = "ui.class_changed"
```

### Event Usage

```python
# Subscribe to events
event_bus.subscribe(Events.DATASET_LOADED, self.on_dataset_loaded)

# Publish events
event_bus.publish(Events.LABELS_CHANGED, {
    'image_path': image_path,
    'box_count': len(boxes)
})
```

## Configuration Management

### Configuration Structure

```python
@dataclass
class AppConfig:
    keymap: KeymapConfig      # Keyboard shortcuts
    ui: UIConfig              # UI preferences
    editor: EditorConfig      # Editor settings
```

### Configuration Files

- **Location**: `~/.yolo-editor/config.json` (Linux/Mac) or `%LOCALAPPDATA%/YOLOEditor/config.json` (Windows)
- **Format**: JSON with support for YAML import/export
- **Auto-save**: Recent files and window state are automatically saved

## Error Handling and Logging

### Logging Levels

- **DEBUG**: Detailed information for debugging
- **INFO**: General information about application flow
- **WARNING**: Potentially harmful situations
- **ERROR**: Error events with exception details

### Log Outputs

- **Console**: User-friendly messages during development
- **File**: Detailed logs with timestamps and context
- **Memory**: In-memory logging for testing

### Error Recovery

Services implement graceful error handling:
- Failed operations return appropriate default values
- Errors are logged with full context
- User is notified of recoverable errors
- Application continues running when possible

## Testing Strategy

### Interface-Based Testing

All services implement interfaces, enabling easy mocking:

```python
class MockDatasetService(IDatasetService):
    def load_dataset(self, path: Path) -> DatasetModel:
        return create_test_dataset()

# Inject mock for testing
container.register_instance(IDatasetService, MockDatasetService())
```

### Service Testing

Each service can be tested in isolation:
- Mock dependencies are injected
- Service behavior is tested independently
- Integration tests verify service interactions

### Event Testing

Event-driven architecture enables testing of component interactions:
- Mock event handlers verify event publishing
- Event history can be inspected for testing
- Components can be tested in isolation

## Performance Optimizations

### Caching Strategy

- **Image Cache**: Recently loaded images (limited size)
- **Label Cache**: Parsed YOLO labels by image path
- **Size Cache**: Image dimensions to avoid repeated loading
- **Dataset Cache**: Resolved dataset structures

### Lazy Loading

- Services are created only when first requested
- Images are loaded on-demand
- Statistics are calculated asynchronously

### Memory Management

- Weak references in event bus prevent memory leaks
- Cache size limits prevent excessive memory usage
- Proper cleanup in service destructors

## Migration Guide

### From Old Architecture

1. **Replace direct service instantiation** with dependency injection
2. **Replace direct method calls** between components with events
3. **Move business logic** from UI classes to service classes
4. **Use configuration service** instead of hardcoded settings

### Example Migration

**Before:**
```python
class MainWindow(QMainWindow):
    def __init__(self):
        self.dataset = None
        
    def load_dataset(self, path):
        self.dataset = resolve_dataset(path)  # Direct call
        self.update_ui()  # Tight coupling
```

**After:**
```python
class MainWindow(QMainWindow):
    def __init__(self, dataset_service: IDatasetService, event_bus: IEventBus):
        self._dataset_service = dataset_service
        self._event_bus = event_bus
        self._event_bus.subscribe(Events.DATASET_LOADED, self.on_dataset_loaded)
        
    def load_dataset(self, path):
        dataset = self._dataset_service.load_dataset(path)  # Service call
        self._event_bus.publish(Events.DATASET_LOADED, dataset)  # Event-driven
```

## Future Enhancements

### Planned Features

1. **Plugin System**: Load additional functionality at runtime
2. **Async Operations**: Background processing for large datasets
3. **Undo/Redo System**: Command pattern for reversible operations
4. **Multi-language Support**: Internationalization framework
5. **Cloud Integration**: Remote dataset and configuration sync

### Extension Points

- **New Services**: Implement service interfaces for new functionality
- **Custom Events**: Define domain-specific events for new features
- **Configuration Extensions**: Add new configuration sections
- **Custom Validators**: Implement validation interfaces for new data types

## Conclusion

The refactored architecture provides:

- **Better Maintainability**: Clear separation of concerns and well-defined interfaces
- **Improved Testability**: Dependency injection and interface-based design
- **Enhanced Extensibility**: Event-driven architecture and service-based design
- **Robust Error Handling**: Comprehensive logging and graceful error recovery
- **Performance Optimization**: Intelligent caching and lazy loading

This architecture supports the current feature set while providing a solid foundation for future enhancements.