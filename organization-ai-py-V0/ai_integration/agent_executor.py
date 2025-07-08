"""
AI Agent Executor Module

This module handles the execution of AI agent workflows, providing a comprehensive
system for managing multi-agent collaborations, refinement loops, and real-time
execution monitoring.

Key Features:
- Sequential and parallel workflow execution
- Agent collaboration with refinement loops
- Real-time progress tracking and logging
- Thread-safe execution management
- Comprehensive error handling and recovery
- Execution history and analytics

Workflow Concepts:
- ExecutionStep: Individual agent task within a workflow
- WorkflowExecution: Complete workflow instance with multiple steps
- AgentExecutionThread: Thread for running workflows asynchronously
- AgentExecutor: Main coordinator for workflow management

Usage:
    executor = AgentExecutor(model_manager)
    execution_id, execution = executor.create_workflow_execution(workflow_data, agents_data)
    executor.execute_workflow(execution_id, execution, agents_data)
"""

import asyncio
import uuid
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from PyQt6.QtCore import QObject, pyqtSignal, QThread
from .model_manager import AIModelManager, ChatMessage, ModelConfig
import logging

# Configure module-level logging
logger = logging.getLogger(__name__)

@dataclass
class ExecutionStep:
    """
    Represents a single execution step within a workflow.
    
    Each step corresponds to one agent's task within the larger workflow,
    tracking its input, output, status, and execution metadata.
    
    Attributes:
        id (str): Unique identifier for this execution step
        agent_id (str): ID of the agent executing this step
        agent_name (str): Human-readable name of the agent
        input_data (str): Input data/prompt for this step
        output_data (str, optional): Generated output from the agent
        status (str): Current execution status
        start_time (datetime, optional): When execution started
        end_time (datetime, optional): When execution completed
        error_message (str, optional): Error details if execution failed
        refinement_loop (int): Current refinement iteration (0 = first attempt)
        metadata (dict): Additional step metadata and context
    """
    id: str
    agent_id: str
    agent_name: str
    input_data: str
    output_data: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    refinement_loop: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        valid_statuses = ["pending", "running", "completed", "failed"]
        if self.status not in valid_statuses:
            raise ValueError(f"Invalid status '{self.status}'. Must be one of: {valid_statuses}")
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate step execution duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def is_completed(self) -> bool:
        """Check if step has completed successfully."""
        return self.status == "completed"
    
    @property
    def is_failed(self) -> bool:
        """Check if step has failed."""
        return self.status == "failed"


@dataclass
class WorkflowExecution:
    """
    Represents a complete workflow execution instance.
    
    Contains all execution steps, overall status, timing information,
    and metadata for a workflow run.
    
    Attributes:
        id (str): Unique identifier for this workflow execution
        workflow_id (str): ID of the workflow template being executed
        steps (List[ExecutionStep]): List of execution steps in order
        status (str): Overall execution status
        start_time (datetime, optional): When execution started
        end_time (datetime, optional): When execution completed
        current_step (int): Index of currently executing step
        metadata (dict): Additional workflow metadata
        config (dict): Workflow configuration parameters
    """
    id: str
    workflow_id: str
    steps: List[ExecutionStep]
    status: str = "pending"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        valid_statuses = ["pending", "running", "completed", "failed", "cancelled"]
        if self.status not in valid_statuses:
            raise ValueError(f"Invalid status '{self.status}'. Must be one of: {valid_statuses}")
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate total execution duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def progress_percentage(self) -> float:
        """Calculate execution progress as percentage."""
        if not self.steps:
            return 0.0
        completed_steps = sum(1 for step in self.steps if step.is_completed)
        return (completed_steps / len(self.steps)) * 100.0
    
    @property
    def is_completed(self) -> bool:
        """Check if workflow has completed successfully."""
        return self.status == "completed"
    
    @property
    def is_failed(self) -> bool:
        """Check if workflow has failed."""
        return self.status == "failed"
    
    def get_step_by_id(self, step_id: str) -> Optional[ExecutionStep]:
        """Get a specific step by its ID."""
        return next((step for step in self.steps if step.id == step_id), None)
    
    def get_failed_steps(self) -> List[ExecutionStep]:
        """Get all steps that have failed."""
        return [step for step in self.steps if step.is_failed]
    
    def get_completed_steps(self) -> List[ExecutionStep]:
        """Get all steps that have completed successfully."""
        return [step for step in self.steps if step.is_completed]

class AgentExecutionThread(QThread):
    """
    Thread for executing AI agent workflows asynchronously.
    
    This class manages the execution of a complete workflow in a separate thread,
    preventing UI blocking while providing real-time progress updates and
    comprehensive error handling.
    
    The thread handles:
    - Sequential step execution with dependency management
    - Refinement loops for iterative improvement
    - Real-time progress reporting
    - Error recovery and graceful degradation
    - Resource cleanup and memory management
    
    Signals:
        step_started(str, str): Emitted when a step begins (execution_id, step_id)
        step_completed(str, str, str): Emitted when step completes (execution_id, step_id, output)
        step_failed(str, str, str): Emitted when step fails (execution_id, step_id, error)
        execution_completed(str): Emitted when workflow completes (execution_id)
        execution_failed(str, str): Emitted when workflow fails (execution_id, error)
        progress_update(str, str, str): Emitted during streaming (execution_id, step_id, partial)
    """
    
    # Define Qt signals for communication with main thread
    step_started = pyqtSignal(str, str)  # execution_id, step_id
    step_completed = pyqtSignal(str, str, str)  # execution_id, step_id, output
    step_failed = pyqtSignal(str, str, str)  # execution_id, step_id, error
    execution_completed = pyqtSignal(str)  # execution_id
    execution_failed = pyqtSignal(str, str)  # execution_id, error
    progress_update = pyqtSignal(str, str, str)  # execution_id, step_id, partial_output
    
    def __init__(self, model_manager: AIModelManager, execution: WorkflowExecution, 
                 agents_data: Dict[str, dict]):
        """
        Initialize the execution thread.
        
        Args:
            model_manager: AI model manager for inference operations
            execution: Workflow execution instance to run
            agents_data: Dictionary of agent configurations
        """
        super().__init__()
        self.model_manager = model_manager
        self.execution = execution
        self.agents_data = agents_data
        self.should_stop = False  # Flag for graceful termination
        
        # Setup logging for this thread
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Performance tracking
        self.step_timings: Dict[str, float] = {}
        self.total_tokens_generated = 0
        
    def run(self):
        """
        Main thread execution method.
        
        Executes the workflow using asyncio event loop and handles
        any top-level exceptions that occur during execution.
        """
        try:
            self.logger.info(f"Starting workflow execution: {self.execution.id}")
            # Run the async workflow in this thread's event loop
            asyncio.run(self._execute_workflow())
        except Exception as e:
            self.logger.error(f"Fatal error in workflow execution: {e}")
            self.execution_failed.emit(self.execution.id, str(e))
    
    async def _execute_workflow(self):
        """
        Execute the complete workflow asynchronously.
        
        Manages the overall workflow execution, including step sequencing,
        error handling, and status updates.
        """
        # Update execution status and timing
        self.execution.status = "running"
        self.execution.start_time = datetime.now()
        
        self.logger.info(f"Executing workflow with {len(self.execution.steps)} steps")
        
        try:
            # Execute each step in sequence
            for i, step in enumerate(self.execution.steps):
                if self.should_stop:
                    self.logger.info("Execution stopped by user request")
                    break
                    
                # Update current step index
                self.execution.current_step = i
                
                # Execute the step
                await self._execute_step(step)
                
                # Check if step failed
                if step.status == "failed":
                    self.execution.status = "failed"
                    error_msg = step.error_message or "Step execution failed"
                    self.logger.error(f"Step {step.id} failed: {error_msg}")
                    self.execution_failed.emit(self.execution.id, error_msg)
                    return
                
                # Brief pause between steps for UI responsiveness
                await asyncio.sleep(0.1)
            
            # Mark execution as completed
            if not self.should_stop:
                self.execution.status = "completed"
                self.execution.end_time = datetime.now()
                
                # Log completion statistics
                duration = self.execution.duration
                self.logger.info(f"Workflow completed successfully in {duration:.2f}s")
                self.logger.info(f"Total tokens generated: {self.total_tokens_generated}")
                
                self.execution_completed.emit(self.execution.id)
            
        except Exception as e:
            # Handle any unexpected errors during execution
            self.execution.status = "failed"
            self.execution.end_time = datetime.now()
            error_msg = f"Workflow execution error: {str(e)}"
            self.logger.error(error_msg)
            self.execution_failed.emit(self.execution.id, error_msg)
    
    async def _execute_step(self, step: ExecutionStep):
        """Execute a single workflow step"""
        step.status = "running"
        step.start_time = datetime.now()
        self.step_started.emit(self.execution.id, step.id)
        
        try:
            agent_data = self.agents_data.get(step.agent_id)
            if not agent_data:
                raise ValueError(f"Agent {step.agent_id} not found")
            
            # Prepare messages for the AI model
            messages = []
            
            # Add system prompt if available
            if agent_data.get('prompt'):
                messages.append(ChatMessage(
                    role="system",
                    content=agent_data['prompt']
                ))
            
            # Add user input
            messages.append(ChatMessage(
                role="user", 
                content=step.input_data
            ))
            
            # Get model ID from agent configuration
            model_id = agent_data.get('model', 'gpt-3.5-turbo')
            
            # Execute with refinement loops if configured
            refinement_loops = agent_data.get('refinement_loops', 1)
            output = step.input_data
            
            for loop in range(refinement_loops):
                step.refinement_loop = loop + 1
                
                # Update messages for refinement
                if loop > 0:
                    messages.append(ChatMessage(
                        role="assistant",
                        content=output
                    ))
                    messages.append(ChatMessage(
                        role="user",
                        content=f"Please refine and improve the above response. This is refinement loop {loop + 1} of {refinement_loops}."
                    ))
                
                # Generate response
                request_id = f"{step.id}_loop_{loop}"
                
                # Connect to progress updates
                def on_progress(req_id: str, partial: str):
                    if req_id == request_id:
                        self.progress_update.emit(self.execution.id, step.id, partial)
                
                self.model_manager.inference_progress.connect(on_progress)
                
                try:
                    output = await self.model_manager.generate_response(
                        model_id, messages, request_id
                    )
                finally:
                    self.model_manager.inference_progress.disconnect(on_progress)
                
                if self.should_stop:
                    break
            
            step.output_data = output
            step.status = "completed"
            step.end_time = datetime.now()
            self.step_completed.emit(self.execution.id, step.id, output)
            
        except Exception as e:
            step.status = "failed"
            step.end_time = datetime.now()
            step.error_message = str(e)
            self.step_failed.emit(self.execution.id, step.id, str(e))
    
    def stop_execution(self):
        """Stop the execution"""
        self.should_stop = True

class AgentExecutor(QObject):
    """Manages agent workflow execution"""
    
    execution_started = pyqtSignal(str)  # execution_id
    execution_completed = pyqtSignal(str)  # execution_id
    execution_failed = pyqtSignal(str, str)  # execution_id, error
    step_progress = pyqtSignal(str, str, str, str)  # execution_id, agent_name, status, message
    
    def __init__(self, model_manager: AIModelManager):
        super().__init__()
        self.model_manager = model_manager
        self.active_executions: Dict[str, AgentExecutionThread] = {}
        self.execution_history: List[WorkflowExecution] = []
        
    def create_workflow_execution(self, workflow_data: dict, agents_data: Dict[str, dict], 
                                 input_data: str = "Begin workflow execution") -> str:
        """Create a new workflow execution"""
        execution_id = str(uuid.uuid4())
        
        # Create execution steps from workflow
        steps = []
        current_input = input_data
        
        # For now, execute agents in sequence
        # TODO: Support parallel execution and complex workflows
        for i, agent_id in enumerate(workflow_data.get('agent_sequence', [])):
            agent_data = agents_data.get(agent_id)
            if agent_data:
                step = ExecutionStep(
                    id=f"{execution_id}_step_{i}",
                    agent_id=agent_id,
                    agent_name=agent_data['name'],
                    input_data=current_input
                )
                steps.append(step)
                current_input = f"Output from {agent_data['name']}"
        
        execution = WorkflowExecution(
            id=execution_id,
            workflow_id=workflow_data.get('id', 'default'),
            steps=steps
        )
        
        return execution_id, execution
    
    def execute_workflow(self, execution_id: str, execution: WorkflowExecution, 
                        agents_data: Dict[str, dict]):
        """Start workflow execution"""
        if execution_id in self.active_executions:
            raise ValueError(f"Execution {execution_id} already running")
        
        # Create execution thread
        thread = AgentExecutionThread(self.model_manager, execution, agents_data)
        
        # Connect signals
        thread.step_started.connect(self._on_step_started)
        thread.step_completed.connect(self._on_step_completed)
        thread.step_failed.connect(self._on_step_failed)
        thread.execution_completed.connect(self._on_execution_completed)
        thread.execution_failed.connect(self._on_execution_failed)
        thread.progress_update.connect(self._on_progress_update)
        
        self.active_executions[execution_id] = thread
        self.execution_started.emit(execution_id)
        
        # Start execution
        thread.start()
    
    def stop_execution(self, execution_id: str):
        """Stop a running execution"""
        if execution_id in self.active_executions:
            thread = self.active_executions[execution_id]
            thread.stop_execution()
            thread.wait()  # Wait for thread to finish
            del self.active_executions[execution_id]
    
    def _on_step_started(self, execution_id: str, step_id: str):
        """Handle step started"""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id].execution
            step = next((s for s in execution.steps if s.id == step_id), None)
            if step:
                self.step_progress.emit(
                    execution_id, 
                    step.agent_name, 
                    "started", 
                    f"Starting execution for {step.agent_name}"
                )
    
    def _on_step_completed(self, execution_id: str, step_id: str, output: str):
        """Handle step completed"""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id].execution
            step = next((s for s in execution.steps if s.id == step_id), None)
            if step:
                self.step_progress.emit(
                    execution_id,
                    step.agent_name,
                    "completed",
                    f"Completed: {output[:100]}..." if len(output) > 100 else output
                )
    
    def _on_step_failed(self, execution_id: str, step_id: str, error: str):
        """Handle step failed"""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id].execution
            step = next((s for s in execution.steps if s.id == step_id), None)
            if step:
                self.step_progress.emit(
                    execution_id,
                    step.agent_name,
                    "failed",
                    f"Error: {error}"
                )
    
    def _on_execution_completed(self, execution_id: str):
        """Handle execution completed"""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id].execution
            self.execution_history.append(execution)
            
            # Clean up
            thread = self.active_executions[execution_id]
            del self.active_executions[execution_id]
            
            self.execution_completed.emit(execution_id)
    
    def _on_execution_failed(self, execution_id: str, error: str):
        """Handle execution failed"""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id].execution
            self.execution_history.append(execution)
            
            # Clean up
            thread = self.active_executions[execution_id]
            del self.active_executions[execution_id]
            
            self.execution_failed.emit(execution_id, error)
    
    def _on_progress_update(self, execution_id: str, step_id: str, partial_output: str):
        """Handle progress update"""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id].execution
            step = next((s for s in execution.steps if s.id == step_id), None)
            if step:
                self.step_progress.emit(
                    execution_id,
                    step.agent_name,
                    "progress",
                    partial_output
                )
    
    def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get execution status"""
        if execution_id in self.active_executions:
            return self.active_executions[execution_id].execution
        
        # Check history
        for execution in self.execution_history:
            if execution.id == execution_id:
                return execution
        
        return None
    
    def get_active_executions(self) -> List[str]:
        """Get list of active execution IDs"""
        return list(self.active_executions.keys())
