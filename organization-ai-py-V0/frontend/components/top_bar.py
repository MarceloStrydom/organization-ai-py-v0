"""
Top Navigation Bar Component

This module provides the main navigation bar for the Organization AI application,
including controls for sidebar toggle, model configuration, and user profile management.
"""

from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QPushButton, 
                            QLabel, QFrame, QSizePolicy)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont, QPixmap
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)


class TopBar(QWidget):
    """
    Top navigation bar widget providing main application controls.
    
    This widget contains the application title, navigation controls, and action buttons
    for model configuration, settings, and user profile management.
    
    Signals:
        sidebar_toggle(): Emitted when sidebar toggle button is clicked
        profile_clicked(): Emitted when profile button is clicked  
        model_config_requested(): Emitted when model configuration is requested
        settings_clicked(): Emitted when settings button is clicked
    """
    
    # Define signals for component communication
    sidebar_toggle = pyqtSignal()
    profile_clicked = pyqtSignal()
    model_config_requested = pyqtSignal()
    settings_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        """
        Initialize the top bar component.
        
        Args:
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.init_ui()
        self.logger.debug("TopBar component initialized")
    
    def init_ui(self):
        """
        Initialize the user interface layout and components.
        
        Creates the main layout with logo, title, and control buttons.
        """
        # Set widget properties
        self.setFixedHeight(64)
        self.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                border-bottom: 1px solid #e5e7eb;
            }
        """)
        
        # Main horizontal layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(16)
        
        # Left side - Brand and navigation
        left_widget = self._create_left_section()
        layout.addWidget(left_widget)
        
        # Spacer to push right content to the edge
        layout.addStretch()
        
        # Right side - Action controls
        right_widget = self._create_right_section()
        layout.addWidget(right_widget)
        
        self.logger.debug("TopBar UI initialization completed")
    
    def _create_left_section(self):
        """
        Create the left section with sidebar toggle and application branding.
        
        Returns:
            QWidget: Left section widget containing navigation controls
        """
        left_widget = QWidget()
        left_layout = QHBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        
        # Sidebar toggle button
        sidebar_btn = self._create_icon_button("☰", "Toggle Sidebar")
        sidebar_btn.clicked.connect(self.sidebar_toggle.emit)
        left_layout.addWidget(sidebar_btn)
        
        # Application title and branding
        title_label = QLabel("🧠 Organization AI")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #111827;
                border: none;
                background: transparent;
            }
        """)
        left_layout.addWidget(title_label)
        
        # Version indicator (optional)
        version_label = QLabel("v0.1")
        version_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #6b7280;
                background-color: #f3f4f6;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 2px 6px;
            }
        """)
        left_layout.addWidget(version_label)
        
        return left_widget
    
    def _create_right_section(self):
        """
        Create the right section with action buttons and controls.
        
        Returns:
            QWidget: Right section widget containing action controls
        """
        right_widget = QWidget()
        right_layout = QHBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        
        # Model configuration button
        model_config_btn = self._create_icon_button("🤖", "Configure AI Models")
        model_config_btn.clicked.connect(self.model_config_requested.emit)
        right_layout.addWidget(model_config_btn)
        
        # Settings button
        settings_btn = self._create_icon_button("⚙️", "Application Settings")
        settings_btn.clicked.connect(self.settings_clicked.emit)
        right_layout.addWidget(settings_btn)
        
        # Add separator
        separator = self._create_separator()
        right_layout.addWidget(separator)
        
        # User profile button
        profile_btn = self._create_icon_button("👤", "User Profile")
        profile_btn.clicked.connect(self.profile_clicked.emit)
        right_layout.addWidget(profile_btn)
        
        return right_widget
    
    def _create_icon_button(self, icon, tooltip):
        """
        Create a standardized icon button with consistent styling.
        
        Args:
            icon (str): Unicode emoji or icon character
            tooltip (str): Tooltip text for the button
            
        Returns:
            QPushButton: Configured icon button
        """
        button = QPushButton(icon)
        button.setFixedSize(36, 36)
        button.setToolTip(tooltip)
        button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 18px;
                font-size: 16px;
                color: #374151;
            }
            QPushButton:hover {
                background-color: #f3f4f6;
                color: #111827;
            }
            QPushButton:pressed {
                background-color: #e5e7eb;
            }
        """)
        return button
    
    def _create_separator(self):
        """
        Create a visual separator for grouping related controls.
        
        Returns:
            QFrame: Vertical separator line
        """
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("""
            QFrame {
                color: #e5e7eb;
                background-color: #e5e7eb;
                border: none;
                max-width: 1px;
            }
        """)
        separator.setFixedHeight(24)
        return separator
    
    def update_user_info(self, username=None, avatar_path=None):
        """
        Update user information display in the top bar.
        
        Args:
            username (str, optional): Username to display
            avatar_path (str, optional): Path to user avatar image
        """
        # This method can be expanded to show user info
        # when user management features are implemented
        self.logger.debug(f"User info updated: {username}")
    
    def set_model_status(self, model_count, active_model=None):
        """
        Update the model configuration button to show current status.
        
        Args:
            model_count (int): Number of configured models
            active_model (str, optional): Name of currently active model
        """
        # Update tooltip to show model information
        tooltip = f"Configure AI Models ({model_count} available)"
        if active_model:
            tooltip += f"\nActive: {active_model}"
        
        # Find and update the model config button
        for child in self.findChildren(QPushButton):
            if child.toolTip().startswith("Configure AI Models"):
                child.setToolTip(tooltip)
                break
