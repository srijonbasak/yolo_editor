from __future__ import annotations
from typing import List, Dict, Tuple, Any
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QPushButton, QHBoxLayout
from PySide6.QtCore import Qt, Signal

class EdgeView(QWidget):
    """Widget for displaying and managing edges/connections in merge designer."""
    
    # Signals
    edge_selected = Signal(tuple, int)  # (dataset_id, class_id), target_id
    edge_removed = Signal(tuple, int)   # (dataset_id, class_id), target_id
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.edges: List[Dict[str, Any]] = []
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        self.title_label = QLabel("Connections")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        header_layout.addWidget(self.title_label)
        
        self.remove_button = QPushButton("Remove Selected")
        self.remove_button.clicked.connect(self._on_remove_selected)
        self.remove_button.setEnabled(False)
        header_layout.addWidget(self.remove_button)
        layout.addLayout(header_layout)
        
        # Edges table
        self.edges_table = QTableWidget(0, 4)
        self.edges_table.setHorizontalHeaderLabels(["Source Dataset", "Source Class", "Target Class", "Status"])
        self.edges_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.edges_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.edges_table.selectionModel().selectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.edges_table)
        
        # Stats
        self.stats_label = QLabel("No connections")
        layout.addWidget(self.stats_label)
    
    def add_edge(self, source_dataset: str, source_class_id: int, source_class_name: str, 
                 target_id: int, target_name: str):
        """Add a new edge/connection."""
        edge = {
            'source_dataset': source_dataset,
            'source_class_id': source_class_id,
            'source_class_name': source_class_name,
            'target_id': target_id,
            'target_name': target_name,
            'status': 'active'
        }
        
        # Check if edge already exists
        for existing in self.edges:
            if (existing['source_dataset'] == source_dataset and 
                existing['source_class_id'] == source_class_id and
                existing['target_id'] == target_id):
                return False  # Edge already exists
        
        self.edges.append(edge)
        self._update_display()
        self.edge_selected.emit((source_dataset, source_class_id), target_id)
        return True
    
    def remove_edge(self, source_dataset: str, source_class_id: int, target_id: int):
        """Remove an edge/connection."""
        for i, edge in enumerate(self.edges):
            if (edge['source_dataset'] == source_dataset and 
                edge['source_class_id'] == source_class_id and
                edge['target_id'] == target_id):
                del self.edges[i]
                self._update_display()
                self.edge_removed.emit((source_dataset, source_class_id), target_id)
                return True
        return False
    
    def get_edges(self) -> List[Dict[str, Any]]:
        """Get all edges."""
        return self.edges.copy()
    
    def get_edges_for_target(self, target_id: int) -> List[Dict[str, Any]]:
        """Get all edges for a specific target."""
        return [edge for edge in self.edges if edge['target_id'] == target_id]
    
    def clear(self):
        """Clear all edges."""
        self.edges.clear()
        self._update_display()
    
    def _update_display(self):
        """Update the display with current edge information."""
        self.edges_table.setRowCount(len(self.edges))
        
        for row, edge in enumerate(self.edges):
            self.edges_table.setItem(row, 0, QTableWidgetItem(edge['source_dataset']))
            self.edges_table.setItem(row, 1, QTableWidgetItem(f"{edge['source_class_name']} ({edge['source_class_id']})"))
            self.edges_table.setItem(row, 2, QTableWidgetItem(f"{edge['target_name']} ({edge['target_id']})"))
            
            status_item = QTableWidgetItem(edge['status'])
            if edge['status'] == 'active':
                status_item.setForeground(Qt.green)
            else:
                status_item.setForeground(Qt.red)
            self.edges_table.setItem(row, 3, status_item)
        
        # Update stats
        if self.edges:
            unique_targets = len(set(edge['target_id'] for edge in self.edges))
            unique_sources = len(set((edge['source_dataset'], edge['source_class_id']) for edge in self.edges))
            self.stats_label.setText(f"{len(self.edges)} connections, {unique_sources} sources → {unique_targets} targets")
        else:
            self.stats_label.setText("No connections")
    
    def _on_selection_changed(self):
        """Handle selection change in edges table."""
        has_selection = len(self.edges_table.selectedItems()) > 0
        self.remove_button.setEnabled(has_selection)
    
    def _on_remove_selected(self):
        """Handle remove selected button click."""
        selected_rows = set()
        for item in self.edges_table.selectedItems():
            selected_rows.add(item.row())
        
        # Remove edges in reverse order to maintain indices
        for row in sorted(selected_rows, reverse=True):
            if 0 <= row < len(self.edges):
                edge = self.edges[row]
                self.remove_edge(edge['source_dataset'], edge['source_class_id'], edge['target_id'])
