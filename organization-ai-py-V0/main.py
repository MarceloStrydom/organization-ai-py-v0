# Add these imports at the top
from ai_integration.model_manager import AIModelManager, get_default_models
from ai_integration.agent_executor import AgentExecutor
from components.model_configuration import ModelConfigurationDialog

class OrganizationAI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.agents = []
        self.models = []
        self.is_executing = False
        self.sidebar_collapsed = False
        self.current_view = "workflow"
        
        # Initialize AI system
        self.model_manager = AIModelManager()
        self.agent_executor = AgentExecutor(self.model_manager)
        
        # Register default models
        for model_config in get_default_models():
            self.model_manager.register_model(model_config)
        
        self.init_ui()
        self.setup_connections()
        
    def setup_connections(self):
        # ... existing connections ...
        
        # AI system connections
        self.model_manager.model_loaded.connect(self.on_model_loaded)
        self.model_manager.model_error.connect(self.on_model_error)
        self.agent_executor.execution_started.connect(self.on_execution_started)
        self.agent_executor.execution_completed.connect(self.on_execution_completed)
        self.agent_executor.execution_failed.connect(self.on_execution_failed)
        
        # Add model configuration menu
        self.top_bar.model_config_requested.connect(self.show_model_configuration)
        
    def show_model_configuration(self):
        """Show model configuration dialog"""
        dialog = ModelConfigurationDialog(self.model_manager, self)
        dialog.exec()
        
    def on_model_loaded(self, model_id):
        """Handle model loaded"""
        print(f"Model loaded: {model_id}")
        
    def on_model_error(self, model_id, error):
        """Handle model error"""
        print(f"Model error for {model_id}: {error}")
        
    def on_execution_started(self, execution_id):
        """Handle execution started"""
        self.is_executing = True
        self.execution_console.set_executing(True)
        
    def on_execution_completed(self, execution_id):
        """Handle execution completed"""
        self.is_executing = False
        self.execution_console.set_executing(False)
        
    def on_execution_failed(self, execution_id, error):
        """Handle execution failed"""
        self.is_executing = False
        self.execution_console.set_executing(False)
        print(f"Execution failed: {error}")
