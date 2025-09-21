from __future__ import annotations
from typing import List, Dict, Any, Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QHBoxLayout, QSpinBox
from PySide6.QtCore import Qt, Signal

class TargetBlock(QWidget):
    """Widget for displaying and managing target classes in merge designer."""
    
    # Signals
    target_added = Signal(str, int)  # name, quota
    target_removed = Signal(int)     # target_id
    quota_changed = Signal(int, int) # target_id, new_quota
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.targets: Dict[int, Dict[str, Any]] = {}
        self._next_id = 0
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        self.title_label = QLabel("Target Classes")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        header_layout.addWidget(self.title_label)
        
        self.add_button = QPushButton("+ Add")
        self.add_button.clicked.connect(self._on_add_target)
        header_layout.addWidget(self.add_button)
        layout.addLayout(header_layout)
        
        # Target list
        self.targets_list = QListWidget()
        self.targets_list.setMaximumHeight(200)
        self.targets_list.itemDoubleClicked.connect(self._on_edit_target)
        layout.addWidget(self.targets_list)
        
        # Stats
        self.stats_label = QLabel("No targets defined")
        layout.addWidget(self.stats_label)
    
    def add_target(self, name: str, quota: Optional[int] = None) -> int:
        """Add a new target class."""
        target_id = self._next_id
        self._next_id += 1
        
        self.targets[target_id] = {
            'id': target_id,
            'name': name,
            'quota': quota,
            'images': 0,
            'boxes': 0
        }
        
        self._update_display()
        self.target_added.emit(name, quota or 0)
        return target_id
    
    def remove_target(self, target_id: int):
        """Remove a target class."""
        if target_id in self.targets:
            del self.targets[target_id]
            self._update_display()
            self.target_removed.emit(target_id)
    
    def update_target_stats(self, target_id: int, images: int, boxes: int):
        """Update statistics for a target class."""
        if target_id in self.targets:
            self.targets[target_id]['images'] = images
            self.targets[target_id]['boxes'] = boxes
            self._update_display()
    
    def set_target_quota(self, target_id: int, quota: int):
        """Set quota for a target class."""
        if target_id in self.targets:
            self.targets[target_id]['quota'] = quota
            self._update_display()
            self.quota_changed.emit(target_id, quota)
    
    def get_targets(self) -> Dict[int, Dict[str, Any]]:
        """Get all target classes."""
        return self.targets.copy()
    
    def _update_display(self):
        """Update the display with current target information."""
        self.targets_list.clear()
        
        for target_id, target_info in sorted(self.targets.items()):
            name = target_info['name']
            quota = target_info['quota']
            images = target_info['images']
            boxes = target_info['boxes']
            
            quota_text = f" (quota: {quota})" if quota else " (unlimited)"
            item_text = f"{name} [{target_id}]{quota_text}\n  {images} images, {boxes} boxes"
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, target_id)
            self.targets_list.addItem(item)
        
        # Update stats
        total_targets = len(self.targets)
        total_images = sum(t['images'] for t in self.targets.values())
        total_boxes = sum(t['boxes'] for t in self.targets.values())
        
        if total_targets > 0:
            self.stats_label.setText(f"{total_targets} targets, {total_images} images, {total_boxes} boxes")
        else:
            self.stats_label.setText("No targets defined")
    
    def _on_add_target(self):
        """Handle add target button click."""
        # This would typically open a dialog, for now just add a default target
        name = f"target_{self._next_id}"
        self.add_target(name)
    
    def _on_edit_target(self, item: QListWidgetItem):
        """Handle double-click on target item."""
        target_id = item.data(Qt.ItemDataRole.UserRole)
        if target_id is not None:
            # This would typically open an edit dialog
            print(f"Edit target {target_id}")
    
    def clear(self):
        """Clear all targets."""
        self.targets.clear()
        self._next_id = 0
        self._update_display()
