from __future__ import annotations
from typing import Dict, Any, Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFormLayout, QLineEdit, QSpinBox, QComboBox, QGroupBox
from PySide6.QtCore import Qt, Signal

class PropertyInspector(QWidget):
    """Widget for inspecting and editing properties of selected items."""
    
    # Signals
    property_changed = Signal(str, Any)  # property_name, new_value
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.current_item: Optional[Dict[str, Any]] = None
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        self.title_label = QLabel("Properties")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.title_label)
        
        # Properties form
        self.properties_group = QGroupBox("Item Properties")
        self.properties_layout = QFormLayout(self.properties_group)
        layout.addWidget(self.properties_group)
        
        # Initially hidden
        self.properties_group.setVisible(False)
    
    def inspect_item(self, item: Dict[str, Any]):
        """Inspect a new item and update the form."""
        self.current_item = item
        self._clear_form()
        self._populate_form(item)
        self.properties_group.setVisible(True)
    
    def clear_inspection(self):
        """Clear the current inspection."""
        self.current_item = None
        self._clear_form()
        self.properties_group.setVisible(False)
    
    def _clear_form(self):
        """Clear all form widgets."""
        while self.properties_layout.count():
            child = self.properties_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def _populate_form(self, item: Dict[str, Any]):
        """Populate the form with item properties."""
        for prop_name, prop_value in item.items():
            if isinstance(prop_value, str):
                widget = QLineEdit(str(prop_value))
                widget.textChanged.connect(lambda text, name=prop_name: self._on_property_changed(name, text))
            elif isinstance(prop_value, int):
                widget = QSpinBox()
                widget.setRange(-999999, 999999)
                widget.setValue(prop_value)
                widget.valueChanged.connect(lambda value, name=prop_name: self._on_property_changed(name, value))
            elif isinstance(prop_value, float):
                widget = QLineEdit(str(prop_value))
                widget.textChanged.connect(lambda text, name=prop_name: self._on_property_changed(name, text))
            elif isinstance(prop_value, bool):
                widget = QComboBox()
                widget.addItems(["False", "True"])
                widget.setCurrentText(str(prop_value))
                widget.currentTextChanged.connect(lambda text, name=prop_name: self._on_property_changed(name, text == "True"))
            else:
                widget = QLineEdit(str(prop_value))
                widget.setReadOnly(True)
            
            self.properties_layout.addRow(prop_name, widget)
    
    def _on_property_changed(self, property_name: str, new_value: Any):
        """Handle property change."""
        if self.current_item is not None:
            self.current_item[property_name] = new_value
            self.property_changed.emit(property_name, new_value)

class DatasetInspector(QWidget):
    """Specialized inspector for dataset properties."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        self.title_label = QLabel("Dataset Inspector")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.title_label)
        
        # Dataset info
        self.info_group = QGroupBox("Dataset Information")
        info_layout = QFormLayout(self.info_group)
        
        self.name_label = QLabel("No dataset selected")
        self.path_label = QLabel("")
        self.classes_label = QLabel("")
        self.images_label = QLabel("")
        
        info_layout.addRow("Name:", self.name_label)
        info_layout.addRow("Path:", self.path_label)
        info_layout.addRow("Classes:", self.classes_label)
        info_layout.addRow("Images:", self.images_label)
        
        layout.addWidget(self.info_group)
        layout.addStretch(1)
    
    def inspect_dataset(self, dataset_info: Dict[str, Any]):
        """Inspect dataset information."""
        self.name_label.setText(dataset_info.get('name', 'Unknown'))
        self.path_label.setText(str(dataset_info.get('path', '')))
        self.classes_label.setText(str(dataset_info.get('num_classes', 0)))
        self.images_label.setText(str(dataset_info.get('num_images', 0)))
    
    def clear_inspection(self):
        """Clear dataset inspection."""
        self.name_label.setText("No dataset selected")
        self.path_label.setText("")
        self.classes_label.setText("")
        self.images_label.setText("")
