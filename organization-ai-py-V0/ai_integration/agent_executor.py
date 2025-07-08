import asyncio
import uuid
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from PyQt6.QtCore import QObject, pyqtSignal, QThread
from .model_manager import AIModelManager, ChatMessage, ModelConfig

@dataclass
class ExecutionStep:
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

@dataclass
class WorkflowExecution:
    id: str
    workflow_id: str
    steps: List[ExecutionStep]
    status: str = "pending"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_step: int = 0

class AgentExecutionThread(QThread):
    """Thread for executing AI agent workflows"""
    
    step_started = pyqtSignal(str, str)  # execution_id, step_id
    step_completed = pyqtSignal(str, str, str)  # execution_id, step_id, output
    step_failed = pyqtSignal(str, str, str)  # execution_id, step_id, error
    execution_completed = pyqtSignal(str)  # execution_id
    execution_failed = pyqtSignal(str, str)  # execution_id, error
    progress_update = pyqtSignal(str, str, str)  # execution_id, step_id, partial_output
    
    def __init__(self, model_manager: AIModelManager, execution: WorkflowExecution, 
                 agents_data: Dict[str, dict]):
        super().__init__()
        self.model_manager = model_manager
        self.execution = execution
        self.agents_data = agents_data
        self.should_stop = False
        
    def run(self):
        """Execute the workflow"""
        try:
            asyncio.run(self._execute_workflow())
        except Exception as e:
            self.execution_failed.emit(self.execution.id, str(e))
    
    async def _execute_workflow(self):
        """Execute workflow steps"""
        self.execution.status = "running"
        self.execution.start_time = datetime.now()
        
        try:
            for i, step in enumerate(self.execution.steps):
                if self.should_stop:
                    break
                    
                self.execution.current_step = i
                await self._execute_step(step)
                
                if step.status == "failed":
                    self.execution.status = "failed"
                    self.execution_failed.emit(self.execution.id, step.error_message or "Step failed")
                    return
            
            self.execution.status = "completed"
            self.execution.end_time = datetime.now()
            self.execution_completed.emit(self.execution.id)
            
        except Exception as e:
            self.execution.status = "failed"
            self.execution.end_time = datetime.now()
            self.execution_failed.emit(self.execution.id, str(e))
    
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
