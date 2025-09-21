from cachetools import LRUCache
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

class ImageCache:
    """Simple LRU cache for QPixmaps or decoded numpy arrays."""
    def __init__(self, max_items: int = 256):
        self.pixmaps = LRUCache(maxsize=max_items)
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached item by key."""
        try:
            return self.pixmaps.get(key)
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        try:
            self.pixmaps[key] = value
        except Exception as e:
            logger.warning(f"Cache put error for key {key}: {e}")
    
    def clear(self) -> None:
        """Clear all cached items."""
        try:
            self.pixmaps.clear()
        except Exception as e:
            logger.warning(f"Cache clear error: {e}")
    
    def __len__(self) -> int:
        """Return number of cached items."""
        return len(self.pixmaps)
