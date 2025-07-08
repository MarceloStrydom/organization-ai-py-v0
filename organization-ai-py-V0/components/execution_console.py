from PyQt5.QtCore import QThread, pyqtSignal as Signal
from datetime import datetime

class ExecutionThread(QThread):
    log_added = Signal(str, str, str)  # timestamp, agent, message
    execution_finished = Signal()
    
    def __init__(self, model_manager, agent_executor, agents_data, parent=None):
        super().__init__(parent)
        self.model_manager = model_manager
        self.agent_executor = agent_executor
        self.agents_data = agents_data
        self.should_stop = False
        self.execution_id = None
        
    def run(self):
        try:
            # Create workflow execution
            workflow_data = {
                'id': 'test_workflow',
                'agent_sequence': list(self.agents_data.keys())
            }
            
            self.execution_id, execution = self.agent_executor.create_workflow_execution(
                workflow_data, self.agents_data, "Please analyze and improve this workflow."
            )
            
            # Connect to progress signals
            self.agent_executor.step_progress.connect(self.on_step_progress)
            self.agent_executor.execution_completed.connect(self.on_execution_completed)
            self.agent_executor.execution_failed.connect(self.on_execution_failed)
            
            # Start execution
            self.agent_executor.execute_workflow(self.execution_id, execution, self.agents_data)
            
        except Exception as e:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_added.emit(timestamp, "System", f"Execution error: {str(e)}")
            self.execution_finished.emit()
    
    def on_step_progress(self, execution_id, agent_name, status, message):
        if execution_id == self.execution_id:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_added.emit(timestamp, agent_name, message)
    
    def on_execution_completed(self, execution_id):
        if execution_id == self.execution_id:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_added.emit(timestamp, "System", "Workflow execution completed successfully")
            self.execution_finished.emit()
    
    def on_execution_failed(self, execution_id, error):
        if execution_id == self.execution_id:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_added.emit(timestamp, "System", f"Execution failed: {error}")
            self.execution_finished.emit()
    
    def stop(self):
        self.should_stop = True
        if self.execution_id:
            self.agent_executor.stop_execution(self.execution_id)
