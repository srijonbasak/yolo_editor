"""
Dependency injection container for YOLO Editor.
Manages service instances and their dependencies.
"""

from __future__ import annotations
from typing import Dict, Type, TypeVar, Callable, Any, Optional, List
from pathlib import Path
import inspect
from dataclasses import dataclass

from .interfaces import (
    IDatasetService, IImageService, IConfigService, ILogger,
    IEventBus, IProgressReporter, IValidationService, IExportService
)
from .dataset_service import DatasetService
from .image_service import ImageService
from .config_service import ConfigService
from .logging_service import LoggingService, LogLevel
from .event_bus import EventBus
from .validation_service import ValidationService

T = TypeVar('T')


@dataclass
class ServiceRegistration:
    """Registration information for a service."""
    service_type: Type
    implementation: Type
    singleton: bool = True
    factory: Optional[Callable] = None
    dependencies: Optional[List[str]] = None


class ServiceContainer:
    """Dependency injection container."""
    
    def __init__(self):
        self._registrations: Dict[str, ServiceRegistration] = {}
        self._instances: Dict[str, Any] = {}
        self._building: set = set()  # Track services being built to detect circular dependencies
    
    def register_singleton(self, service_type: Type[T], implementation: Type[T]) -> 'ServiceContainer':
        """Register a service as a singleton."""
        key = self._get_service_key(service_type)
        self._registrations[key] = ServiceRegistration(
            service_type=service_type,
            implementation=implementation,
            singleton=True
        )
        return self
    
    def register_transient(self, service_type: Type[T], implementation: Type[T]) -> 'ServiceContainer':
        """Register a service as transient (new instance each time)."""
        key = self._get_service_key(service_type)
        self._registrations[key] = ServiceRegistration(
            service_type=service_type,
            implementation=implementation,
            singleton=False
        )
        return self
    
    def register_factory(self, service_type: Type[T], factory: Callable[[], T], 
                        singleton: bool = True) -> 'ServiceContainer':
        """Register a service with a factory function."""
        key = self._get_service_key(service_type)
        self._registrations[key] = ServiceRegistration(
            service_type=service_type,
            implementation=None,
            singleton=singleton,
            factory=factory
        )
        return self
    
    def register_instance(self, service_type: Type[T], instance: T) -> 'ServiceContainer':
        """Register a specific instance."""
        key = self._get_service_key(service_type)
        self._instances[key] = instance
        return self
    
    def get(self, service_type: Type[T]) -> T:
        """Get a service instance."""
        key = self._get_service_key(service_type)
        
        # Check if we already have an instance
        if key in self._instances:
            return self._instances[key]
        
        # Check if service is registered
        if key not in self._registrations:
            raise ValueError(f"Service {service_type.__name__} is not registered")
        
        # Check for circular dependencies
        if key in self._building:
            raise ValueError(f"Circular dependency detected for service {service_type.__name__}")
        
        registration = self._registrations[key]
        
        try:
            self._building.add(key)
            
            # Create instance
            if registration.factory:
                instance = self._create_from_factory(registration)
            else:
                instance = self._create_from_type(registration)
            
            # Store if singleton
            if registration.singleton:
                self._instances[key] = instance
            
            return instance
            
        finally:
            self._building.discard(key)
    
    def _create_from_factory(self, registration: ServiceRegistration) -> Any:
        """Create instance using factory function."""
        factory = registration.factory
        
        # Check if factory needs dependencies
        sig = inspect.signature(factory)
        if sig.parameters:
            # Factory has parameters, try to inject dependencies
            kwargs = {}
            for param_name, param in sig.parameters.items():
                if param.annotation != inspect.Parameter.empty:
                    dependency = self.get(param.annotation)
                    kwargs[param_name] = dependency
            return factory(**kwargs)
        else:
            return factory()
    
    def _create_from_type(self, registration: ServiceRegistration) -> Any:
        """Create instance from type with dependency injection."""
        impl_type = registration.implementation
        
        # Get constructor signature
        sig = inspect.signature(impl_type.__init__)
        kwargs = {}
        
        # Inject dependencies
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            if param.annotation != inspect.Parameter.empty:
                try:
                    dependency = self.get(param.annotation)
                    kwargs[param_name] = dependency
                except ValueError:
                    # Dependency not registered, check if parameter has default
                    if param.default == inspect.Parameter.empty:
                        raise ValueError(
                            f"Cannot resolve dependency {param.annotation.__name__} "
                            f"for {impl_type.__name__}.{param_name}"
                        )
        
        return impl_type(**kwargs)
    
    def _get_service_key(self, service_type: Type) -> str:
        """Get a unique key for a service type."""
        return f"{service_type.__module__}.{service_type.__name__}"
    
    def is_registered(self, service_type: Type) -> bool:
        """Check if a service type is registered."""
        key = self._get_service_key(service_type)
        return key in self._registrations or key in self._instances
    
    def clear(self) -> None:
        """Clear all registrations and instances."""
        self._registrations.clear()
        self._instances.clear()
        self._building.clear()
    
    def get_registered_services(self) -> List[str]:
        """Get list of registered service names."""
        return list(self._registrations.keys()) + list(self._instances.keys())


class ServiceContainerBuilder:
    """Builder for configuring the service container."""
    
    def __init__(self):
        self._container = ServiceContainer()
    
    def configure_default_services(self, log_file: Optional[Path] = None) -> 'ServiceContainerBuilder':
        """Configure default services."""
        
        # Logger (must be first as others depend on it)
        self._container.register_factory(
            ILogger, 
            lambda: LoggingService("YOLOEditor", log_file, LogLevel.INFO, LogLevel.DEBUG)
        )
        
        # Event bus
        self._container.register_singleton(IEventBus, EventBus)
        
        # Configuration service
        self._container.register_singleton(IConfigService, ConfigService)
        
        # Image service
        self._container.register_singleton(IImageService, ImageService)
        
        # Dataset service
        self._container.register_singleton(IDatasetService, DatasetService)
        
        # Validation service
        self._container.register_singleton(IValidationService, ValidationService)
        
        return self
    
    def configure_logging(self, log_file: Optional[Path] = None, 
                         console_level: LogLevel = LogLevel.INFO,
                         file_level: LogLevel = LogLevel.DEBUG) -> 'ServiceContainerBuilder':
        """Configure logging service."""
        self._container.register_factory(
            ILogger,
            lambda: LoggingService("YOLOEditor", log_file, console_level, file_level)
        )
        return self
    
    def configure_config(self, config_dir: Optional[Path] = None) -> 'ServiceContainerBuilder':
        """Configure configuration service."""
        if config_dir:
            self._container.register_factory(
                IConfigService,
                lambda logger: ConfigService(logger, config_dir)
            )
        else:
            self._container.register_singleton(IConfigService, ConfigService)
        return self
    
    def add_custom_service(self, service_type: Type[T], implementation: Type[T], 
                          singleton: bool = True) -> 'ServiceContainerBuilder':
        """Add a custom service."""
        if singleton:
            self._container.register_singleton(service_type, implementation)
        else:
            self._container.register_transient(service_type, implementation)
        return self
    
    def add_instance(self, service_type: Type[T], instance: T) -> 'ServiceContainerBuilder':
        """Add a service instance."""
        self._container.register_instance(service_type, instance)
        return self
    
    def build(self) -> ServiceContainer:
        """Build and return the configured container."""
        return self._container


# Global container instance
_global_container: Optional[ServiceContainer] = None


def get_container() -> ServiceContainer:
    """Get the global service container."""
    global _global_container
    if _global_container is None:
        _global_container = (ServiceContainerBuilder()
                           .configure_default_services()
                           .build())
    return _global_container


def set_container(container: ServiceContainer) -> None:
    """Set the global service container."""
    global _global_container
    _global_container = container


def get_service(service_type: Type[T]) -> T:
    """Get a service from the global container."""
    return get_container().get(service_type)


def configure_services(log_file: Optional[Path] = None, 
                      config_dir: Optional[Path] = None) -> ServiceContainer:
    """Configure and set up the global service container."""
    builder = ServiceContainerBuilder()
    
    if log_file:
        builder.configure_logging(log_file)
    
    if config_dir:
        builder.configure_config(config_dir)
    
    container = builder.configure_default_services().build()
    set_container(container)
    
    return container


# Decorator for automatic dependency injection
def inject(func: Callable) -> Callable:
    """Decorator to automatically inject dependencies into function parameters."""
    sig = inspect.signature(func)
    
    def wrapper(*args, **kwargs):
        # Get the container
        container = get_container()
        
        # Inject missing dependencies
        bound_args = sig.bind_partial(*args, **kwargs)
        
        for param_name, param in sig.parameters.items():
            if param_name not in bound_args.arguments and param.annotation != inspect.Parameter.empty:
                try:
                    dependency = container.get(param.annotation)
                    bound_args.arguments[param_name] = dependency
                except ValueError:
                    # Dependency not available, check if parameter has default
                    if param.default == inspect.Parameter.empty:
                        raise ValueError(f"Cannot inject dependency {param.annotation.__name__}")
        
        return func(*bound_args.args, **bound_args.kwargs)
    
    return wrapper