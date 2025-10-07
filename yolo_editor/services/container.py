"""
Dependency injection container for YOLO Editor.
Manages service instances and their dependencies.
"""

from __future__ import annotations

from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    ForwardRef,
)
from pathlib import Path
import inspect
import sys
from dataclasses import dataclass

from .interfaces import (
    IDatasetService,
    IImageService,
    IConfigService,
    ILogger,
    IEventBus,
    IProgressReporter,
    IValidationService,
    IExportService,
)
from .dataset_service import DatasetService
from .image_service import ImageService
from .config_service import ConfigService
from .logging_service import LoggingService, LogLevel
from .event_bus import EventBus
from .validation_service import ValidationService

T = TypeVar("T")


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
        self._building: set[str] = set()  # Track services being built to detect circular dependencies

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------
    def register_singleton(self, service_type: Type[T], implementation: Type[T]) -> "ServiceContainer":
        key = self._get_service_key(service_type)
        self._registrations[key] = ServiceRegistration(
            service_type=service_type,
            implementation=implementation,
            singleton=True,
        )
        return self

    def register_transient(self, service_type: Type[T], implementation: Type[T]) -> "ServiceContainer":
        key = self._get_service_key(service_type)
        self._registrations[key] = ServiceRegistration(
            service_type=service_type,
            implementation=implementation,
            singleton=False,
        )
        return self

    def register_factory(
        self,
        service_type: Type[T],
        factory: Callable[..., T],
        singleton: bool = True,
    ) -> "ServiceContainer":
        key = self._get_service_key(service_type)
        self._registrations[key] = ServiceRegistration(
            service_type=service_type,
            implementation=None,
            singleton=singleton,
            factory=factory,
        )
        return self

    def register_instance(self, service_type: Type[T], instance: T) -> "ServiceContainer":
        key = self._get_service_key(service_type)
        self._instances[key] = instance
        return self

    # ------------------------------------------------------------------
    # Resolution API
    # ------------------------------------------------------------------
    def get(self, service_type: Type[T]) -> T:
        key = self._get_service_key(service_type)

        if key in self._instances:
            return self._instances[key]

        if key not in self._registrations:
            raise ValueError(f"Service {service_type.__name__} is not registered")

        if key in self._building:
            raise ValueError(f"Circular dependency detected for service {service_type.__name__}")

        registration = self._registrations[key]

        try:
            self._building.add(key)
            if registration.factory:
                instance = self._create_from_factory(registration)
            else:
                instance = self._create_from_type(registration)
            if registration.singleton:
                self._instances[key] = instance
            return instance
        finally:
            self._building.discard(key)

    # ------------------------------------------------------------------
    # Instance creation helpers
    # ------------------------------------------------------------------
    def _create_from_factory(self, registration: ServiceRegistration) -> Any:
        factory = registration.factory
        sig = inspect.signature(factory)
        globalns = getattr(factory, "__globals__", {}) or {}
        try:
            type_hints = get_type_hints(factory, globalns=globalns, localns=None)
        except Exception:
            type_hints = {}

        if not sig.parameters:
            return factory()

        kwargs: Dict[str, Any] = {}
        for param_name, param in sig.parameters.items():
            annotation = type_hints.get(param_name, param.annotation)
            dependency_type = self._resolve_annotation(annotation, globalns)
            if dependency_type is None:
                continue
            if not self.is_registered(dependency_type):
                if param.default == inspect.Parameter.empty:
                    raise ValueError(
                        f"Cannot resolve dependency {getattr(dependency_type, '__name__', dependency_type)!r} "
                        f"for factory {factory.__name__}.{param_name}"
                    )
                continue
            kwargs[param_name] = self.get(dependency_type)
        return factory(**kwargs)

    def _create_from_type(self, registration: ServiceRegistration) -> Any:
        impl_type = registration.implementation
        sig = inspect.signature(impl_type.__init__)
        module = sys.modules.get(impl_type.__module__)
        globalns = vars(module) if module else {}
        try:
            type_hints = get_type_hints(impl_type.__init__, globalns=globalns, localns=None)
        except Exception:
            type_hints = {}

        kwargs: Dict[str, Any] = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            annotation = type_hints.get(param_name, param.annotation)
            dependency_type = self._resolve_annotation(annotation, globalns)
            if dependency_type is None:
                continue
            if not self.is_registered(dependency_type):
                if param.default == inspect.Parameter.empty:
                    raise ValueError(
                        f"Cannot resolve dependency {dependency_type.__name__} for "
                        f"{impl_type.__name__}.{param_name}"
                    )
                continue
            kwargs[param_name] = self.get(dependency_type)

        return impl_type(**kwargs)

    def _resolve_annotation(self, annotation: Any, globalns: Optional[Dict[str, Any]]) -> Optional[Type]:
        """Resolve postponed / composite annotations into a concrete type."""

        if annotation is inspect.Parameter.empty or annotation is None or annotation is Any:
            return None

        if isinstance(annotation, ForwardRef):
            try:
                evaluated = eval(annotation.__forward_arg__, globalns or {}, {})
            except Exception:
                return None
            return self._resolve_annotation(evaluated, globalns)

        if isinstance(annotation, str):
            try:
                evaluated = eval(annotation, globalns or {}, {})
            except Exception:
                return None
            return self._resolve_annotation(evaluated, globalns)

        origin = get_origin(annotation)
        if origin is None:
            return annotation if isinstance(annotation, type) else None

        if origin in (list, dict, tuple, set):
            return None

        if origin is Annotated:
            base, *_ = get_args(annotation)
            return self._resolve_annotation(base, globalns)

        if origin is Union:
            resolved = [
                self._resolve_annotation(arg, globalns)
                for arg in get_args(annotation)
                if arg is not type(None)  # noqa: E721
            ]
            resolved = [arg for arg in resolved if arg is not None]
            if len(resolved) == 1:
                return resolved[0]
            return None

        return None

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------
    def _get_service_key(self, service_type: Type) -> str:
        if not hasattr(service_type, "__module__") or not hasattr(service_type, "__name__"):
            raise TypeError(f"Service key expects a type, got {service_type!r}")
        return f"{service_type.__module__}.{service_type.__name__}"

    def is_registered(self, service_type: Type) -> bool:
        try:
            key = self._get_service_key(service_type)
        except TypeError:
            return False
        return key in self._registrations or key in self._instances

    def clear(self) -> None:
        self._registrations.clear()
        self._instances.clear()
        self._building.clear()

    def get_registered_services(self) -> List[str]:
        return list(self._registrations.keys()) + list(self._instances.keys())


class ServiceContainerBuilder:
    """Builder for configuring the service container."""

    def __init__(self):
        self._container = ServiceContainer()

    def configure_default_services(self, log_file: Optional[Path] = None) -> "ServiceContainerBuilder":
        self._container.register_factory(
            ILogger,
            lambda: LoggingService("YOLOEditor", log_file, LogLevel.INFO, LogLevel.DEBUG),
        )
        self._container.register_singleton(IEventBus, EventBus)
        self._container.register_singleton(IConfigService, ConfigService)
        self._container.register_singleton(IImageService, ImageService)
        self._container.register_singleton(IDatasetService, DatasetService)
        self._container.register_singleton(IValidationService, ValidationService)
        return self

    def configure_logging(
        self,
        log_file: Optional[Path] = None,
        console_level: LogLevel = LogLevel.INFO,
        file_level: LogLevel = LogLevel.DEBUG,
    ) -> "ServiceContainerBuilder":
        self._container.register_factory(
            ILogger,
            lambda: LoggingService("YOLOEditor", log_file, console_level, file_level),
        )
        return self

    def configure_config(self, config_dir: Optional[Path] = None) -> "ServiceContainerBuilder":
        if config_dir:
            def config_factory(logger: ILogger) -> IConfigService:
                return ConfigService(logger, config_dir)

            self._container.register_factory(IConfigService, config_factory)
        else:
            self._container.register_singleton(IConfigService, ConfigService)
        return self

    def add_custom_service(
        self,
        service_type: Type[T],
        implementation: Type[T],
        singleton: bool = True,
    ) -> "ServiceContainerBuilder":
        if singleton:
            self._container.register_singleton(service_type, implementation)
        else:
            self._container.register_transient(service_type, implementation)
        return self

    def add_instance(self, service_type: Type[T], instance: T) -> "ServiceContainerBuilder":
        self._container.register_instance(service_type, instance)
        return self

    def build(self) -> ServiceContainer:
        return self._container


# Global container instance
_global_container: Optional[ServiceContainer] = None


def get_container() -> ServiceContainer:
    global _global_container
    if _global_container is None:
        _global_container = ServiceContainerBuilder().configure_default_services().build()
    return _global_container


def set_container(container: ServiceContainer) -> None:
    global _global_container
    _global_container = container


def get_service(service_type: Type[T]) -> T:
    return get_container().get(service_type)


def configure_services(
    log_file: Optional[Path] = None,
    config_dir: Optional[Path] = None,
) -> ServiceContainer:
    builder = ServiceContainerBuilder()
    if log_file:
        builder.configure_logging(log_file)
    if config_dir:
        builder.configure_config(config_dir)
    container = builder.configure_default_services().build()
    set_container(container)
    return container


def inject(func: Callable) -> Callable:
    """Decorator to automatically inject dependencies into function parameters."""

    sig = inspect.signature(func)
    globalns = getattr(func, "__globals__", {}) or {}
    try:
        type_hints = get_type_hints(func, globalns=globalns, localns=None)
    except Exception:
        type_hints = {}

    def wrapper(*args, **kwargs):
        container = get_container()
        bound_args = sig.bind_partial(*args, **kwargs)

        for param_name, param in sig.parameters.items():
            if param_name in bound_args.arguments:
                continue
            annotation = type_hints.get(param_name, param.annotation)
            dependency_type = container._resolve_annotation(annotation, globalns)
            if dependency_type is None:
                continue
            if not container.is_registered(dependency_type):
                if param.default == inspect.Parameter.empty:
                    raise ValueError(
                        f"Cannot inject dependency {dependency_type.__name__} "
                        f"for parameter '{param_name}' in {func.__name__}"
                    )
                continue
            bound_args.arguments[param_name] = container.get(dependency_type)

        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper
