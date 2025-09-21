from __future__ import annotations
from typing import List, Dict, Any
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget, QListWidgetItem
from PySide6.QtCore import Qt

class DatasetBlock(QWidget):
    """Widget for displaying dataset information in merge designer."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.dataset_info: Dict[str, Any] = {}
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        self.title_label = QLabel("Dataset")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.title_label)
        
        self.classes_list = QListWidget()
        self.classes_list.setMaximumHeight(150)
        layout.addWidget(self.classes_list)
        
        self.stats_label = QLabel("No dataset loaded")
        layout.addWidget(self.stats_label)
    
    def set_dataset_info(self, dataset_id: str, classes: List[Dict[str, Any]]):
        """Set dataset information and update display."""
        self.dataset_info = {
            'id': dataset_id,
            'classes': classes
        }
        
        self.title_label.setText(f"Dataset: {dataset_id}")
        
        # Clear and populate classes list
        self.classes_list.clear()
        for cls in classes:
            item_text = f"{cls.get('name', 'Unknown')} (ID: {cls.get('id', '?')})"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, cls)
            self.classes_list.addItem(item)
        
        # Update stats
        total_images = sum(cls.get('images', 0) for cls in classes)
        total_boxes = sum(cls.get('boxes', 0) for cls in classes)
        self.stats_label.setText(f"Total: {total_images} images, {total_boxes} boxes")
    
    def get_selected_classes(self) -> List[Dict[str, Any]]:
        """Get currently selected classes."""
        selected = []
        for item in self.classes_list.selectedItems():
            cls_data = item.data(Qt.ItemDataRole.UserRole)
            if cls_data:
                selected.append(cls_data)
        return selected
    
    def clear(self):
        """Clear all dataset information."""
        self.dataset_info = {}
        self.title_label.setText("Dataset")
        self.classes_list.clear()
        self.stats_label.setText("No dataset loaded")
