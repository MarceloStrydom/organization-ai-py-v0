class TopBar(QWidget):
    sidebar_toggle = pyqtSignal()
    profile_clicked = pyqtSignal()
    model_config_requested = pyqtSignal()  # Add this signal
    
    def init_ui(self):
        # ... existing code ...
        
        # Right side - Controls
        right_widget = QWidget()
        right_layout = QHBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        
        # Model configuration button
        model_config_btn = QPushButton("🤖")
        model_config_btn.setFixedSize(32, 32)
        model_config_btn.setToolTip("Configure AI Models")
        model_config_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 16px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #f3f4f6;
            }
        """)
        model_config_btn.clicked.connect(self.model_config_requested.emit)
        right_layout.addWidget(model_config_btn)
        
        # ... rest of existing buttons ...
