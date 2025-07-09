"""
Execution Console Module

This module handles the execution of AI agent workflows in a separate thread,
providing real-time progress monitoring and logging capabilities.
"""

from PyQt6.QtCore import QThread, pyqtSignal as Signal
from datetime import datetime
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)

class ExecutionThread(QThread):
    """
    Thread for executing AI agent workflows.
    
    This class runs agent workflows in a separate thread to prevent UI blocking,
    providing real-time progress updates and execution status monitoring.
    
    Signals:
        log_added(str, str, str): Emitted when a new log entry is added (timestamp, agent, message)
        execution_finished(): Emitted when workflow execution completes or fails
    """
    log_added = Signal(str, str, str)  # timestamp, agent, message
    execution_finished = Signal()
    
    def __init__(self, model_manager, agent_executor, agents_data, parent=None):
        """
        Initialize the execution thread.
        
        Args:
            model_manager: AI model manager instance for model operations
            agent_executor: Agent executor instance for workflow management  
            agents_data: Dictionary containing agent configuration data
            parent: Parent QObject (optional)
        """
        super().__init__(parent)
        self.model_manager = model_manager
        self.agent_executor = agent_executor
        self.agents_data = agents_data
        self.should_stop = False  # Flag for graceful thread termination
        self.execution_id = None  # Current execution identifier
        
        # Set up logging for this thread
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def run(self):
        """
        Main thread execution method.
        
        Creates and executes a workflow from the provided agent data,
        handling progress monitoring and error reporting.
        """
        try:
            # Create workflow execution configuration
            workflow_data = {
                'id': 'test_workflow',
                'agent_sequence': list(self.agents_data.keys())
            }
            
            # Log workflow initiation
            self.logger.info(f"Starting workflow execution with {len(self.agents_data)} agents")
            
            # Create workflow execution instance
            self.execution_id, execution = self.agent_executor.create_workflow_execution(
                workflow_data, self.agents_data, "Please analyze and improve this workflow."
            )
            
            # Connect to progress signals for real-time updates
            self.agent_executor.step_progress.connect(self.on_step_progress)
            self.agent_executor.execution_completed.connect(self.on_execution_completed)
            self.agent_executor.execution_failed.connect(self.on_execution_failed)
            
            # Start workflow execution
            self.agent_executor.execute_workflow(self.execution_id, execution, self.agents_data)
            
        except Exception as e:
            # Handle and log execution errors
            self.logger.error(f"Workflow execution error: {str(e)}")
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_added.emit(timestamp, "System", f"Execution error: {str(e)}")
            self.execution_finished.emit()
    
    def on_step_progress(self, execution_id, agent_name, status, message):
        """
        Handle step progress updates from agent execution.
        
        Args:
            execution_id: Unique identifier for the execution
            agent_name: Name of the agent currently executing
            status: Current status of the step
            message: Progress message to display
        """
        if execution_id == self.execution_id:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_added.emit(timestamp, agent_name, message)
    
    def on_execution_completed(self, execution_id):
        """
        Handle successful completion of workflow execution.
        
        Args:
            execution_id: Unique identifier for the completed execution
        """
        if execution_id == self.execution_id:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_added.emit(timestamp, "System", "Workflow execution completed successfully")
            self.logger.info(f"Workflow execution {execution_id} completed successfully")
            self.execution_finished.emit()
    
    def on_execution_failed(self, execution_id, error):
        """
        Handle failed workflow execution.
        
        Args:
            execution_id: Unique identifier for the failed execution
            error: Error message describing the failure
        """
        if execution_id == self.execution_id:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_added.emit(timestamp, "System", f"Execution failed: {error}")
            self.logger.error(f"Workflow execution {execution_id} failed: {error}")
            self.execution_finished.emit()
    
    def stop(self):
        """
        Gracefully stop the execution thread.
        
        Sets the stop flag and instructs the agent executor to halt the current execution.
        This method ensures proper cleanup and resource disposal.
        """
        self.should_stop = True
        if self.execution_id:
            self.logger.info(f"Stopping execution {self.execution_id}")
            self.agent_executor.stop_execution(self.execution_id)
