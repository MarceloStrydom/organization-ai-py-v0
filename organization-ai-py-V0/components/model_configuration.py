from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QLineEdit, QPushButton, QTabWidget, QWidget,
                            QFormLayout, QComboBox, QSpinBox, QTextEdit,
                            QMessageBox, QGroupBox, QCheckBox)
from PyQt6.QtCore import Qt, pyqtSignal
from ai_integration.model_manager import AIModelManager, ModelType, ModelConfig, get_default_models

class APIKeyDialog(QDialog):
    """Dialog for configuring API keys"""
    
    keys_updated = pyqtSignal(dict)
    
    def __init__(self, current_keys: dict, parent=None):
        super().__init__(parent)
        self.current_keys = current_keys.copy()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Configure API Keys")
        self.setModal(True)
        self.resize(500, 400)
        
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel(
            "Configure your API keys to use cloud-based AI models. "
            "Keys are stored securely in your local configuration."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #6b7280; margin-bottom: 16px;")
        layout.addWidget(instructions)
        
        # API Key inputs
        form_layout = QFormLayout()
        
        self.api_inputs = {}
        
        # OpenAI
        openai_group = QGroupBox("OpenAI")
        openai_layout = QFormLayout(openai_group)
        
        self.api_inputs['openai'] = QLineEdit()
        self.api_inputs['openai'].setPlaceholderText("sk-...")
        self.api_inputs['openai'].setText(self.current_keys.get('openai', ''))
        self.api_inputs['openai'].setEchoMode(QLineEdit.EchoMode.Password)
        openai_layout.addRow("API Key:", self.api_inputs['openai'])
        
        openai_show = QCheckBox("Show key")
        openai_show.toggled.connect(
            lambda checked: self.api_inputs['openai'].setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        openai_layout.addRow("", openai_show)
        
        layout.addWidget(openai_group)
        
        # Anthropic
        anthropic_group = QGroupBox("Anthropic")
        anthropic_layout = QFormLayout(anthropic_group)
        
        self.api_inputs['anthropic'] = QLineEdit()
        self.api_inputs['anthropic'].setPlaceholderText("sk-ant-...")
        self.api_inputs['anthropic'].setText(self.current_keys.get('anthropic', ''))
        self.api_inputs['anthropic'].setEchoMode(QLineEdit.EchoMode.Password)
        anthropic_layout.addRow("API Key:", self.api_inputs['anthropic'])
        
        anthropic_show = QCheckBox("Show key")
        anthropic_show.toggled.connect(
            lambda checked: self.api_inputs['anthropic'].setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        anthropic_layout.addRow("", anthropic_show)
        
        layout.addWidget(anthropic_group)
        
        # Groq
        groq_group = QGroupBox("Groq")
        groq_layout = QFormLayout(groq_group)
        
        self.api_inputs['groq'] = QLineEdit()
        self.api_inputs['groq'].setPlaceholderText("gsk_...")
        self.api_inputs['groq'].setText(self.current_keys.get('groq', ''))
        self.api_inputs['groq'].setEchoMode(QLineEdit.EchoMode.Password)
        groq_layout.addRow("API Key:", self.api_inputs['groq'])
        
        groq_show = QCheckBox("Show key")
        groq_show.toggled.connect(
            lambda checked: self.api_inputs['groq'].setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        groq_layout.addRow("", groq_show)
        
        layout.addWidget(groq_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        save_btn = QPushButton("💾 Save Keys")
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)
        save_btn.clicked.connect(self.save_keys)
        
        cancel_btn = QPushButton("❌ Cancel")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #f3f4f6;
                color: #374151;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #e5e7eb;
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
    def save_keys(self):
        """Save API keys"""
        keys = {}
        for provider, input_field in self.api_inputs.items():
            key = input_field.text().strip()
            if key:
                keys[provider] = key
        
        self.keys_updated.emit(keys)
        self.accept()

class ModelConfigurationDialog(QDialog):
    """Dialog for configuring AI models"""
    
    def __init__(self, model_manager: AIModelManager, parent=None):
        super().__init__(parent)
        self.model_manager = model_manager
        self.init_ui()
        self.load_models()
        
    def init_ui(self):
        self.setWindowTitle("AI Model Configuration")
        self.setModal(True)
        self.resize(800, 600)
        
        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        
        title = QLabel("AI Model Configuration")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #111827;")
        header_layout.addWidget(title)
        
        # API Keys button
        api_keys_btn = QPushButton("🔑 Configure API Keys")
        api_keys_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)
        api_keys_btn.clicked.connect(self.configure_api_keys)
        header_layout.addWidget(api_keys_btn)
        
        layout.addLayout(header_layout)
        
        # Tabs for different model types
        self.tabs = QTabWidget()
        
        # API Models tab
        api_tab = QWidget()
        self.setup_api_models_tab(api_tab)
        self.tabs.addTab(api_tab, "🌐 API Models")
        
        # Local Models tab
        local_tab = QWidget()
        self.setup_local_models_tab(local_tab)
        self.tabs.addTab(local_tab, "💻 Local Models")
        
        layout.addWidget(self.tabs)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        test_btn = QPushButton("🧪 Test Selected Model")
        test_btn.clicked.connect(self.test_model)
        
        close_btn = QPushButton("✅ Close")
        close_btn.clicked.connect(self.accept)
        
        button_layout.addWidget(test_btn)
        button_layout.addStretch()
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
    def setup_api_models_tab(self, tab):
        """Setup API models tab"""
        layout = QVBoxLayout(tab)
        
        # Instructions
        instructions = QLabel(
            "API models require internet connection and valid API keys. "
            "They offer the best performance but have usage costs."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #6b7280; margin-bottom: 16px;")
        layout.addWidget(instructions)
        
        # Model list will be populated dynamically
        self.api_models_layout = QVBoxLayout()
        layout.addLayout(self.api_models_layout)
        
        layout.addStretch()
        
    def setup_local_models_tab(self, tab):
        """Setup local models tab"""
        layout = QVBoxLayout(tab)
        
        # Instructions
        instructions = QLabel(
            "Local models run on your computer without internet connection. "
            "They may require significant disk space and processing power."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #6b7280; margin-bottom: 16px;")
        layout.addWidget(instructions)
        
        # HuggingFace section
        hf_group = QGroupBox("🤗 HuggingFace Models")
        hf_layout = QVBoxLayout(hf_group)
        
        hf_instructions = QLabel(
            "Enter a HuggingFace model path (e.g., 'microsoft/DialoGPT-medium')"
        )
        hf_instructions.setStyleSheet("color: #6b7280; margin-bottom: 8px;")
        hf_layout.addWidget(hf_instructions)
        
        hf_input_layout = QHBoxLayout()
        self.hf_model_input = QLineEdit()
        self.hf_model_input.setPlaceholderText("microsoft/DialoGPT-medium")
        
        add_hf_btn = QPushButton("➕ Add Model")
        add_hf_btn.clicked.connect(self.add_huggingface_model)
        
        hf_input_layout.addWidget(self.hf_model_input)
        hf_input_layout.addWidget(add_hf_btn)
        hf_layout.addLayout(hf_input_layout)
        
        layout.addWidget(hf_group)
        
        # Ollama section
        ollama_group = QGroupBox("🦙 Ollama Models")
        ollama_layout = QVBoxLayout(ollama_group)
        
        ollama_instructions = QLabel(
            "Ollama must be installed and running. Enter model name (e.g., 'llama2', 'mistral')"
        )
        ollama_instructions.setStyleSheet("color: #6b7280; margin-bottom: 8px;")
        ollama_layout.addWidget(ollama_instructions)
        
        ollama_input_layout = QHBoxLayout()
        self.ollama_model_input = QLineEdit()
        self.ollama_model_input.setPlaceholderText("llama2")
        
        self.ollama_endpoint_input = QLineEdit()
        self.ollama_endpoint_input.setPlaceholderText("http://localhost:11434")
        self.ollama_endpoint_input.setText("http://localhost:11434")
        
        add_ollama_btn = QPushButton("➕ Add Model")
        add_ollama_btn.clicked.connect(self.add_ollama_model)
        
        ollama_input_layout.addWidget(QLabel("Model:"))
        ollama_input_layout.addWidget(self.ollama_model_input)
        ollama_input_layout.addWidget(QLabel("Endpoint:"))
        ollama_input_layout.addWidget(self.ollama_endpoint_input)
        ollama_input_layout.addWidget(add_ollama_btn)
        ollama_layout.addLayout(ollama_input_layout)
        
        layout.addWidget(ollama_group)
        
        # Local models list
        self.local_models_layout = QVBoxLayout()
        layout.addLayout(self.local_models_layout)
        
        layout.addStretch()
        
    def load_models(self):
        """Load and display available models"""
        # Register default models
        for model_config in get_default_models():
            self.model_manager.register_model(model_config)
        
        # Display models
        self.refresh_model_lists()
        
    def refresh_model_lists(self):
        """Refresh model lists in tabs"""
        # Clear existing layouts
        self.clear_layout(self.api_models_layout)
        self.clear_layout(self.local_models_layout)
        
        # Get models
        models = self.model_manager.get_available_models()
        
        # Separate API and local models
        api_models = [m for m in models if m.type in [ModelType.API_OPENAI, ModelType.API_ANTHROPIC, ModelType.API_GROQ]]
        local_models = [m for m in models if m.type in [ModelType.LOCAL_HUGGINGFACE, ModelType.LOCAL_OLLAMA]]
        
        # Display API models
        for model in api_models:
            self.add_model_widget(model, self.api_models_layout)
            
        # Display local models
        for model in local_models:
            self.add_model_widget(model, self.local_models_layout)
    
    def clear_layout(self, layout):
        """Clear all widgets from layout"""
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def add_model_widget(self, model: ModelConfig, parent_layout):
        """Add a model widget to the layout"""
        widget = QWidget()
        widget.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 16px;
                margin: 4px 0;
            }
        """)
        
        layout = QHBoxLayout(widget)
        
        # Model info
        info_layout = QVBoxLayout()
        
        name_label = QLabel(model.name)
        name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        type_label = QLabel(f"Type: {model.type.value}")
        type_label.setStyleSheet("color: #6b7280; font-size: 12px;")
        
        if model.model_path:
            path_label = QLabel(f"Path: {model.model_path}")
            path_label.setStyleSheet("color: #6b7280; font-size: 12px;")
            info_layout.addWidget(path_label)
        
        info_layout.addWidget(name_label)
        info_layout.addWidget(type_label)
        
        layout.addLayout(info_layout)
        layout.addStretch()
        
        # Actions
        if model.type in [ModelType.LOCAL_HUGGINGFACE, ModelType.LOCAL_OLLAMA]:
            load_btn = QPushButton("📥 Load")
            load_btn.clicked.connect(lambda: self.load_model(model.id))
            layout.addWidget(load_btn)
        
        test_btn = QPushButton("🧪 Test")
        test_btn.clicked.connect(lambda: self.test_specific_model(model.id))
        layout.addWidget(test_btn)
        
        parent_layout.addWidget(widget)
    
    def configure_api_keys(self):
        """Open API keys configuration dialog"""
        dialog = APIKeyDialog(self.model_manager.api_keys, self)
        dialog.keys_updated.connect(self.model_manager.save_api_keys)
        dialog.exec()
    
    def add_huggingface_model(self):
        """Add a HuggingFace model"""
        model_path = self.hf_model_input.text().strip()
        if not model_path:
            QMessageBox.warning(self, "Error", "Please enter a model path")
            return
        
        model_id = model_path.replace('/', '-').lower()
        model_name = model_path.split('/')[-1] if '/' in model_path else model_path
        
        config = ModelConfig(
            id=model_id,
            name=f"{model_name} (HuggingFace)",
            type=ModelType.LOCAL_HUGGINGFACE,
            model_path=model_path
        )
        
        self.model_manager.register_model(config)
        self.hf_model_input.clear()
        self.refresh_model_lists()
        
        QMessageBox.information(self, "Success", f"Added model: {model_name}")
    
    def add_ollama_model(self):
        """Add an Ollama model"""
        model_name = self.ollama_model_input.text().strip()
        endpoint = self.ollama_endpoint_input.text().strip()
        
        if not model_name:
            QMessageBox.warning(self, "Error", "Please enter a model name")
            return
        
        model_id = f"ollama-{model_name}"
        
        config = ModelConfig(
            id=model_id,
            name=f"{model_name} (Ollama)",
            type=ModelType.LOCAL_OLLAMA,
            model_path=model_name,
            endpoint=endpoint
        )
        
        self.model_manager.register_model(config)
        self.ollama_model_input.clear()
        self.refresh_model_lists()
        
        QMessageBox.information(self, "Success", f"Added model: {model_name}")
    
    def load_model(self, model_id: str):
        """Load a local model"""
        success = self.model_manager.load_model(model_id)
        if success:
            QMessageBox.information(self, "Success", f"Model {model_id} loaded successfully")
        else:
            QMessageBox.warning(self, "Error", f"Failed to load model {model_id}")
    
    def test_model(self):
        """Test the currently selected model"""
        # For now, test the first available model
        models = self.model_manager.get_available_models()
        if models:
            self.test_specific_model(models[0].id)
        else:
            QMessageBox.warning(self, "Error", "No models available")
    
    def test_specific_model(self, model_id: str):
        """Test a specific model"""
        QMessageBox.information(
            self, 
            "Model Test", 
            f"Testing model {model_id}...\n\n"
            "This would send a test prompt to the model and display the response. "
            "Implementation depends on the specific model type and configuration."
        )
