"""
Event bus implementation for YOLO Editor.
Provides decoupled communication between components using the observer pattern.
"""

from __future__ import annotations
from typing import Dict, List, Callable, Any, Optional
from collections import defaultdict
import weakref
from dataclasses import dataclass
from datetime import datetime

from .interfaces import IEventBus, ILogger


@dataclass
class EventData:
    """Container for event data."""
    event_type: str
    data: Any
    timestamp: datetime
    source: Optional[str] = None


class EventBus(IEventBus):
    """Concrete implementation of event bus."""
    
    def __init__(self, logger: ILogger):
        self._logger = logger
        self._handlers: Dict[str, List[weakref.WeakMethod]] = defaultdict(list)
        self._event_history: List[EventData] = []
        self._max_history = 1000
        self._enabled = True
    
    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to an event type."""
        if not callable(handler):
            self._logger.error(f"Handler for event '{event_type}' is not callable")
            return
        
        try:
            # Use weak references to prevent memory leaks
            if hasattr(handler, '__self__'):
                # Bound method
                weak_handler = weakref.WeakMethod(handler, self._cleanup_handler)
            else:
                # Function - we'll store it directly but this is less common
                weak_handler = weakref.ref(handler, self._cleanup_handler)
            
            self._handlers[event_type].append(weak_handler)
            self._logger.debug(f"Subscribed to event '{event_type}'", handler=str(handler))
            
        except Exception as e:
            self._logger.error(f"Failed to subscribe to event '{event_type}'", exception=e)
    
    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe from an event type."""
        if event_type not in self._handlers:
            return
        
        try:
            handlers_to_remove = []
            for weak_handler in self._handlers[event_type]:
                actual_handler = weak_handler()
                if actual_handler is None or actual_handler == handler:
                    handlers_to_remove.append(weak_handler)
            
            for weak_handler in handlers_to_remove:
                self._handlers[event_type].remove(weak_handler)
            
            # Clean up empty event type lists
            if not self._handlers[event_type]:
                del self._handlers[event_type]
            
            self._logger.debug(f"Unsubscribed from event '{event_type}'", handler=str(handler))
            
        except Exception as e:
            self._logger.error(f"Failed to unsubscribe from event '{event_type}'", exception=e)
    
    def publish(self, event_type: str, data: Any = None) -> None:
        """Publish an event."""
        if not self._enabled:
            return
        
        try:
            event_data = EventData(
                event_type=event_type,
                data=data,
                timestamp=datetime.now()
            )
            
            # Add to history
            self._add_to_history(event_data)
            
            # Get handlers for this event type
            handlers = self._handlers.get(event_type, [])
            if not handlers:
                self._logger.debug(f"No handlers for event '{event_type}'")
                return
            
            # Call all handlers
            dead_handlers = []
            for weak_handler in handlers:
                handler = weak_handler()
                if handler is None:
                    # Handler was garbage collected
                    dead_handlers.append(weak_handler)
                    continue
                
                try:
                    # Call the handler
                    if data is not None:
                        handler(data)
                    else:
                        handler()
                    
                except Exception as e:
                    self._logger.error(
                        f"Error in event handler for '{event_type}'", 
                        exception=e, 
                        handler=str(handler)
                    )
            
            # Clean up dead handlers
            for dead_handler in dead_handlers:
                handlers.remove(dead_handler)
            
            self._logger.debug(f"Published event '{event_type}' to {len(handlers)} handlers")
            
        except Exception as e:
            self._logger.error(f"Failed to publish event '{event_type}'", exception=e)
    
    def _cleanup_handler(self, weak_ref) -> None:
        """Clean up dead weak references."""
        for event_type, handlers in list(self._handlers.items()):
            if weak_ref in handlers:
                handlers.remove(weak_ref)
                if not handlers:
                    del self._handlers[event_type]
                break
    
    def _add_to_history(self, event_data: EventData) -> None:
        """Add event to history."""
        self._event_history.append(event_data)
        
        # Trim history if it gets too long
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]
    
    def get_event_history(self, event_type: Optional[str] = None, 
                         limit: Optional[int] = None) -> List[EventData]:
        """Get event history, optionally filtered by type and limited."""
        history = self._event_history
        
        if event_type:
            history = [e for e in history if e.event_type == event_type]
        
        if limit:
            history = history[-limit:]
        
        return history.copy()
    
    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()
        self._logger.debug("Event history cleared")
    
    def get_subscriber_count(self, event_type: Optional[str] = None) -> Dict[str, int]:
        """Get count of subscribers for each event type."""
        if event_type:
            return {event_type: len(self._handlers.get(event_type, []))}
        
        return {et: len(handlers) for et, handlers in self._handlers.items()}
    
    def enable(self) -> None:
        """Enable event publishing."""
        self._enabled = True
        self._logger.debug("Event bus enabled")
    
    def disable(self) -> None:
        """Disable event publishing."""
        self._enabled = False
        self._logger.debug("Event bus disabled")
    
    def is_enabled(self) -> bool:
        """Check if event bus is enabled."""
        return self._enabled
    
    def publish_async(self, event_type: str, data: Any = None) -> None:
        """Publish an event asynchronously (for future implementation)."""
        # For now, just publish synchronously
        # In the future, this could use QTimer.singleShot or similar
        self.publish(event_type, data)
    
    def create_scoped_publisher(self, source: str) -> 'ScopedEventPublisher':
        """Create a scoped publisher that adds source information to events."""
        return ScopedEventPublisher(self, source)


class ScopedEventPublisher:
    """Event publisher that automatically adds source information."""
    
    def __init__(self, event_bus: EventBus, source: str):
        self._event_bus = event_bus
        self._source = source
    
    def publish(self, event_type: str, data: Any = None) -> None:
        """Publish an event with source information."""
        if isinstance(data, dict):
            data = dict(data)  # Copy to avoid modifying original
            data['_source'] = self._source
        else:
            data = {'data': data, '_source': self._source}
        
        self._event_bus.publish(event_type, data)
    
    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to an event (delegates to main event bus)."""
        self._event_bus.subscribe(event_type, handler)
    
    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe from an event (delegates to main event bus)."""
        self._event_bus.unsubscribe(event_type, handler)


class EventBusDecorator:
    """Decorator for automatically publishing events from method calls."""
    
    def __init__(self, event_bus: EventBus, event_type: str, 
                 include_args: bool = False, include_result: bool = False):
        self._event_bus = event_bus
        self._event_type = event_type
        self._include_args = include_args
        self._include_result = include_result
    
    def __call__(self, func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Prepare event data
            event_data = {'method': func.__name__}
            
            if self._include_args:
                event_data['args'] = args[1:]  # Skip 'self'
                event_data['kwargs'] = kwargs
            
            try:
                result = func(*args, **kwargs)
                
                if self._include_result:
                    event_data['result'] = result
                
                event_data['success'] = True
                self._event_bus.publish(self._event_type, event_data)
                
                return result
                
            except Exception as e:
                event_data['success'] = False
                event_data['error'] = str(e)
                self._event_bus.publish(self._event_type, event_data)
                raise
        
        return wrapper