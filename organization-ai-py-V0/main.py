#!/usr/bin/env python3
"""
Organization AI - Main Application Module

A modern desktop application for building and executing intelligent AI workflows
using visual drag-and-drop interface with configurable AI agents and models.

This application enables users to:
- Design agent-based automations visually  
- Configure AI models (OpenAI, Anthropic, Groq, HuggingFace, Ollama)
- Execute workflows with real-time monitoring
- Manage model downloads and configurations
- Run entirely offline with local models

Author: Organization AI Team
Version: 0.1.0
License: MIT
"""

import sys
import os
import logging
from typing import List, Dict, Optional

# PyQt6 imports for GUI components
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QSplitter, QFrame, QMessageBox,
                            QStatusBar, QMenuBar, QMenu, QToolBar, QSystemTrayIcon)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread, QSettings
from PyQt6.QtGui import QAction, QIcon, QPixmap, QFont

# Import AI integration modules
from ai_integration.model_manager import AIModelManager, get_default_models
from ai_integration.agent_executor import AgentExecutor

# Import UI components
from components.model_configuration import ModelConfigurationDialog
from components.top_bar import TopBar
from components.execution_console import ExecutionThread

# Import utilities
from utils.theme import apply_dark_theme, get_color_scheme

# Configure application-wide logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('organization_ai.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class OrganizationAI(QMainWindow):
    """
    Main application window for Organization AI.
    
    This class serves as the primary container for all application functionality,
    managing the user interface, AI models, agent execution, and workflow management.
    
    Key Features:
    - Visual workflow builder with drag-and-drop interface
    - AI model management (local and cloud-based)
    - Real-time execution monitoring and logging
    - Agent collaboration with refinement loops
    - Offline operation capability
    """
    
    def __init__(self):
        """Initialize the main application window."""
        super().__init__()
        
        # Application state variables
        self.agents = []  # List of configured agents
        self.models = []  # List of available AI models
        self.is_executing = False  # Workflow execution status
        self.sidebar_collapsed = False  # Sidebar visibility state
        self.current_view = "workflow"  # Active view mode
        self.settings = QSettings("OrganizationAI", "MainApp")  # Application settings
        
        # Initialize AI system components
        self.model_manager = AIModelManager()
        self.agent_executor = AgentExecutor(self.model_manager)
        
        # UI components will be initialized in init_ui()
        self.top_bar = None
        self.execution_console = None
        self.status_bar = None
        self.sidebar = None
        self.main_content = None
        
        # Setup logging for this instance
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("Initializing Organization AI application")
        
        # Initialize the application
        self._register_default_models()
        self.init_ui()
        self.setup_connections()
        self.restore_settings()
        
        self.logger.info("Organization AI application initialized successfully")
        
    def _register_default_models(self):
        """Register default AI models for immediate availability."""
        try:
            default_models = get_default_models()
            for model_config in default_models:
                self.model_manager.register_model(model_config)
            self.logger.info(f"Registered {len(default_models)} default AI models")
        except Exception as e:
            self.logger.error(f"Failed to register default models: {e}")
            
    def init_ui(self):
        """
        Initialize the user interface layout and components.
        
        Creates the main window layout with top bar, sidebar, content area,
        and status bar following modern desktop application patterns.
        """
        self.logger.debug("Initializing user interface")
        
        # Set main window properties
        self.setWindowTitle("🧠 Organization AI - Intelligent Workflow Builder")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(800, 600)
        
        # Apply modern theme
        apply_dark_theme(self)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create and add top navigation bar
        self.top_bar = TopBar()
        main_layout.addWidget(self.top_bar)
        
        # Create main content area with splitter
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(content_splitter)
        
        # Create sidebar for navigation and tools
        self.sidebar = self._create_sidebar()
        content_splitter.addWidget(self.sidebar)
        
        # Create main content area
        self.main_content = self._create_main_content()
        content_splitter.addWidget(self.main_content)
        
        # Set splitter proportions (sidebar: 20%, content: 80%)
        content_splitter.setSizes([280, 1120])
        
        # Create status bar
        self.status_bar = self._create_status_bar()
        self.setStatusBar(self.status_bar)
        
        # Create menu bar
        self._create_menu_bar()
        
        self.logger.debug("User interface initialization completed")
        
    def _create_sidebar(self):
        """
        Create the sidebar widget with navigation and tool panels.
        
        Returns:
            QWidget: Configured sidebar widget
        """
        sidebar = QFrame()
        sidebar.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #f9fafb;
                border-right: 1px solid #e5e7eb;
                min-width: 250px;
                max-width: 400px;
            }
        """)
        
        layout = QVBoxLayout(sidebar)
        
        # Sidebar content will be added in future iterations
        # For now, add placeholder
        from PyQt6.QtWidgets import QLabel
        placeholder = QLabel("Sidebar\n(Agents & Tools)")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: #6b7280; font-size: 14px;")
        layout.addWidget(placeholder)
        
        return sidebar
        
    def _create_main_content(self):
        """
        Create the main content area widget.
        
        Returns:
            QWidget: Configured main content widget
        """
        content = QFrame()
        content.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: none;
            }
        """)
        
        layout = QVBoxLayout(content)
        
        # Main content will be workflow canvas, execution console, etc.
        # For now, add placeholder
        from PyQt6.QtWidgets import QLabel
        placeholder = QLabel("Main Content Area\n(Workflow Canvas)")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: #6b7280; font-size: 16px;")
        layout.addWidget(placeholder)
        
        return content
        
    def _create_status_bar(self):
        """
        Create the status bar with application status indicators.
        
        Returns:
            QStatusBar: Configured status bar
        """
        status_bar = QStatusBar()
        status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #f9fafb;
                border-top: 1px solid #e5e7eb;
                color: #374151;
            }
        """)
        
        # Add status indicators
        status_bar.showMessage("Ready - No active executions")
        
        return status_bar
        
    def _create_menu_bar(self):
        """Create the application menu bar with standard menus."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        new_workflow_action = QAction('&New Workflow', self)
        new_workflow_action.setShortcut('Ctrl+N')
        new_workflow_action.triggered.connect(self._new_workflow)
        file_menu.addAction(new_workflow_action)
        
        open_workflow_action = QAction('&Open Workflow', self)
        open_workflow_action.setShortcut('Ctrl+O')
        open_workflow_action.triggered.connect(self._open_workflow)
        file_menu.addAction(open_workflow_action)
        
        save_workflow_action = QAction('&Save Workflow', self)
        save_workflow_action.setShortcut('Ctrl+S')
        save_workflow_action.triggered.connect(self._save_workflow)
        file_menu.addAction(save_workflow_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('&Tools')
        
        model_config_action = QAction('&Model Configuration', self)
        model_config_action.triggered.connect(self.show_model_configuration)
        tools_menu.addAction(model_config_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        about_action = QAction('&About', self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def setup_connections(self):
        """
        Set up signal-slot connections between components.
        
        Connects UI components to their respective handlers and establishes
        communication between AI system components and the user interface.
        """
        self.logger.debug("Setting up component connections")
        
        # Top bar connections
        if self.top_bar:
            self.top_bar.sidebar_toggle.connect(self._toggle_sidebar)
            self.top_bar.profile_clicked.connect(self._show_profile)
            self.top_bar.model_config_requested.connect(self.show_model_configuration)
            self.top_bar.settings_clicked.connect(self._show_settings)
        
        # AI system connections
        self.model_manager.model_loaded.connect(self.on_model_loaded)
        self.model_manager.model_error.connect(self.on_model_error)
        self.agent_executor.execution_started.connect(self.on_execution_started)
        self.agent_executor.execution_completed.connect(self.on_execution_completed)
        self.agent_executor.execution_failed.connect(self.on_execution_failed)
        
        self.logger.debug("Component connections established")
        
    def show_model_configuration(self):
        """
        Show the model configuration dialog.
        
        Opens a modal dialog allowing users to configure AI models,
        manage API keys, and test model connections.
        """
        self.logger.info("Opening model configuration dialog")
        try:
            dialog = ModelConfigurationDialog(self.model_manager, self)
            result = dialog.exec()
            if result:
                self.logger.info("Model configuration dialog completed successfully")
            else:
                self.logger.info("Model configuration dialog cancelled")
        except Exception as e:
            self.logger.error(f"Error opening model configuration: {e}")
            QMessageBox.critical(self, "Error", f"Failed to open model configuration: {e}")
        
    def on_model_loaded(self, model_id):
        """
        Handle successful model loading.
        
        Args:
            model_id (str): Identifier of the loaded model
        """
        self.logger.info(f"Model loaded successfully: {model_id}")
        if self.status_bar:
            self.status_bar.showMessage(f"Model loaded: {model_id}", 3000)
        
    def on_model_error(self, model_id, error):
        """
        Handle model loading errors.
        
        Args:
            model_id (str): Identifier of the model that failed to load
            error (str): Error message describing the failure
        """
        self.logger.error(f"Model error for {model_id}: {error}")
        if self.status_bar:
            self.status_bar.showMessage(f"Model error: {model_id}", 5000)
        QMessageBox.warning(self, "Model Error", f"Error with model {model_id}: {error}")
        
    def on_execution_started(self, execution_id):
        """
        Handle workflow execution start.
        
        Args:
            execution_id (str): Unique identifier for the execution
        """
        self.logger.info(f"Workflow execution started: {execution_id}")
        self.is_executing = True
        if self.status_bar:
            self.status_bar.showMessage(f"Executing workflow: {execution_id}")
        
        # Update execution console if available
        if hasattr(self, 'execution_console') and self.execution_console:
            self.execution_console.set_executing(True)
        
    def on_execution_completed(self, execution_id):
        """
        Handle successful workflow execution completion.
        
        Args:
            execution_id (str): Unique identifier for the completed execution
        """
        self.logger.info(f"Workflow execution completed: {execution_id}")
        self.is_executing = False
        if self.status_bar:
            self.status_bar.showMessage("Workflow execution completed successfully", 3000)
        
        # Update execution console if available
        if hasattr(self, 'execution_console') and self.execution_console:
            self.execution_console.set_executing(False)
        
    def on_execution_failed(self, execution_id, error):
        """
        Handle workflow execution failure.
        
        Args:
            execution_id (str): Unique identifier for the failed execution
            error (str): Error message describing the failure
        """
        self.logger.error(f"Workflow execution failed: {execution_id} - {error}")
        self.is_executing = False
        if self.status_bar:
            self.status_bar.showMessage(f"Execution failed: {error}", 5000)
        
        # Update execution console if available
        if hasattr(self, 'execution_console') and self.execution_console:
            self.execution_console.set_executing(False)
        
        # Show error dialog for critical failures
        QMessageBox.critical(self, "Execution Failed", f"Workflow execution failed: {error}")

    # UI Event Handlers
    def _toggle_sidebar(self):
        """Toggle sidebar visibility."""
        self.sidebar_collapsed = not self.sidebar_collapsed
        if self.sidebar:
            self.sidebar.setVisible(not self.sidebar_collapsed)
        self.logger.debug(f"Sidebar toggled: {'collapsed' if self.sidebar_collapsed else 'expanded'}")
        
    def _show_profile(self):
        """Show user profile dialog."""
        QMessageBox.information(self, "Profile", "User profile management coming soon!")
        
    def _show_settings(self):
        """Show application settings dialog."""
        QMessageBox.information(self, "Settings", "Application settings coming soon!")
        
    def _new_workflow(self):
        """Create a new workflow."""
        QMessageBox.information(self, "New Workflow", "New workflow creation coming soon!")
        
    def _open_workflow(self):
        """Open an existing workflow."""
        QMessageBox.information(self, "Open Workflow", "Workflow loading coming soon!")
        
    def _save_workflow(self):
        """Save the current workflow."""
        QMessageBox.information(self, "Save Workflow", "Workflow saving coming soon!")
        
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(self, "About Organization AI", 
                         "Organization AI v0.1.0\n\n"
                         "A modern application for building and executing "
                         "intelligent AI workflows with visual interface.\n\n"
                         "Features offline operation with local models and "
                         "cloud-based AI integration.")

    def restore_settings(self):
        """Restore application settings from previous session."""
        try:
            # Restore window geometry
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
                
            # Restore window state
            state = self.settings.value("windowState")
            if state:
                self.restoreState(state)
                
            # Restore sidebar state
            sidebar_collapsed = self.settings.value("sidebarCollapsed", False, type=bool)
            if sidebar_collapsed != self.sidebar_collapsed:
                self._toggle_sidebar()
                
            self.logger.debug("Application settings restored")
        except Exception as e:
            self.logger.warning(f"Failed to restore settings: {e}")
            
    def save_settings(self):
        """Save current application settings."""
        try:
            self.settings.setValue("geometry", self.saveGeometry())
            self.settings.setValue("windowState", self.saveState())
            self.settings.setValue("sidebarCollapsed", self.sidebar_collapsed)
            self.logger.debug("Application settings saved")
        except Exception as e:
            self.logger.warning(f"Failed to save settings: {e}")
            
    def closeEvent(self, event):
        """
        Handle application close event.
        
        Ensures proper cleanup and saves application state before closing.
        
        Args:
            event: Close event object
        """
        self.logger.info("Application closing - performing cleanup")
        
        # Save application settings
        self.save_settings()
        
        # Stop any running executions
        if self.is_executing:
            reply = QMessageBox.question(self, 'Execution in Progress',
                                       'A workflow is currently executing. Stop and exit?',
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                       QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            
            # Stop active executions
            try:
                active_executions = self.agent_executor.get_active_executions()
                for execution_id in active_executions:
                    self.agent_executor.stop_execution(execution_id)
                self.logger.info(f"Stopped {len(active_executions)} active executions")
            except Exception as e:
                self.logger.error(f"Error stopping executions: {e}")
        
        # Accept the close event
        event.accept()
        self.logger.info("Application closed successfully")


def main():
    """
    Main application entry point.
    
    Initializes the QApplication, creates the main window, and starts the event loop.
    Handles command line arguments and application-level configuration.
    """
    # Create QApplication instance
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("Organization AI")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("OrganizationAI")
    app.setOrganizationDomain("organizationai.com")
    
    # Set application icon (if available)
    try:
        app.setWindowIcon(QIcon("icon.png"))
    except:
        pass  # Icon file not found, continue without icon
    
    # Apply application-wide font
    font = QFont("Inter", 10)
    app.setFont(font)
    
    # Create and show main window
    logger.info("Starting Organization AI application")
    
    try:
        main_window = OrganizationAI()
        main_window.show()
        
        logger.info("Application started successfully")
        
        # Start the event loop
        exit_code = app.exec()
        
        logger.info(f"Application exited with code: {exit_code}")
        return exit_code
        
    except Exception as e:
        logger.critical(f"Fatal error starting application: {e}")
        QMessageBox.critical(None, "Fatal Error", 
                           f"Failed to start Organization AI:\n{e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
