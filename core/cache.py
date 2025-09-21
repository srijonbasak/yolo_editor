from cachetools import LRUCache

class ImageCache:
    """Simple LRU cache for QPixmaps or decoded numpy arrays."""
    def __init__(self, max_items: int = 256):
        self.pixmaps = LRUCache(maxsize=max_items)
