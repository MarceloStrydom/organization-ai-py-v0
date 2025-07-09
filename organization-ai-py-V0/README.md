# Organization AI - Python Desktop Application

A modern, feature-rich desktop application for building and executing intelligent AI workflows using a visual drag-and-drop interface with configurable AI agents and models.

## 🚀 Features

- 🧠 **Multi-AI Model Support** - OpenAI, Anthropic, Groq, HuggingFace, Ollama
- 🎨 **Visual Workflow Builder** - Drag-and-drop interface for creating AI workflows
- 💻 **Offline Operation** - Complete local model support with caching
- 🔄 **Agent Collaboration** - Multi-agent workflows with refinement loops
- 📊 **Real-time Monitoring** - Live execution tracking and progress updates
- 🎯 **Modern UI** - Professional PyQt6 desktop interface with dark/light themes
- ⚙️ **Configuration Management** - Easy setup and model configuration
- 🔐 **Secure API Key Management** - Encrypted storage for cloud AI services

## 📋 Requirements

- **Python 3.8+** (Recommended: Python 3.9 or higher)
- **Operating System**: Windows, macOS, or Linux
- **Memory**: At least 4GB RAM (8GB+ recommended for local models)
- **Disk Space**: 2GB+ free space (more needed for local model downloads)

## 🔧 Installation

### Option 1: Quick Install (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MarceloStrydom/organization-ai-py-v0.git
   cd organization-ai-py-v0/organization-ai-py-V0
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

### Option 2: Virtual Environment (Recommended for Development)

1. **Clone and navigate:**
   ```bash
   git clone https://github.com/MarceloStrydom/organization-ai-py-v0.git
   cd organization-ai-py-v0/organization-ai-py-V0
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python main.py
   ```

## 🧪 Testing

Run the comprehensive test suite to verify installation:

```bash
python tests/run_tests.py
```

All tests should pass before running the main application.

## 🏃‍♂️ Quick Start

1. **First Launch**: The application will start with default AI model configurations
2. **Configure Models**: Go to Tools → Model Configuration to set up your API keys
3. **Create Workflow**: Use File → New Workflow to start building your AI workflow
4. **Add Agents**: Configure AI agents with different models and prompts
5. **Execute**: Run your workflow and monitor real-time progress

## 🛠️ Configuration

### AI Model Setup

The application supports multiple AI providers:

#### Cloud Models (Require API Keys)
- **OpenAI**: GPT-3.5, GPT-4 models
- **Anthropic**: Claude 3 family
- **Groq**: Mixtral, Llama 2 models

#### Local Models (No API Key Required)
- **HuggingFace**: Download and run models locally
- **Ollama**: Local model serving (requires Ollama installation)

### API Key Configuration

1. Open Tools → Model Configuration
2. Select your AI provider
3. Enter your API key
4. Test the connection
5. Save configuration

API keys are stored securely in your user configuration directory.

## 📁 Project Structure

```
organization-ai-py-V0/
├── main.py                    # Application entry point
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── .gitignore                # Git ignore rules
├── config/                   # Configuration management
│   ├── __init__.py
│   └── app_config.py         # Application configuration
├── core/                     # Core data structures and utilities
│   ├── __init__.py
│   ├── models/               # Data models and classes
│   │   ├── __init__.py
│   │   └── data_models.py    # Core data structures
│   └── utils/                # Core utilities
│       ├── __init__.py
│       └── helpers.py        # Helper functions
├── backend/                  # Backend services and AI integration
│   ├── __init__.py
│   ├── ai/                   # AI model management
│   │   ├── __init__.py
│   │   ├── model_manager.py  # AI model loading and management
│   │   └── agent_executor.py # Agent execution and workflows
│   └── services/             # Backend services
│       └── __init__.py
├── frontend/                 # Frontend UI components
│   ├── __init__.py
│   ├── components/           # UI components
│   │   ├── __init__.py
│   │   ├── top_bar.py       # Top navigation bar
│   │   ├── model_configuration.py # Model setup dialog
│   │   └── execution_console.py # Execution monitoring
│   └── ui/                   # UI utilities and theming
│       ├── __init__.py
│       └── theme.py         # UI theming and styling
├── data/                     # Application data storage
│   └── README.md            # Data directory documentation
└── tests/                    # Test suite
    ├── __init__.py
    └── run_tests.py         # Main test runner
```

## 🔍 Troubleshooting

### Common Issues

**PyQt6 Installation Issues:**
```bash
# On Ubuntu/Debian:
sudo apt-get install libegl1 libgl1-mesa-dev

# On macOS with Homebrew:
brew install qt6

# On Windows: Usually works out of the box
```

**Missing System Dependencies:**
```bash
# Ubuntu/Debian:
sudo apt-get install python3-dev build-essential

# CentOS/RHEL:
sudo yum install python3-devel gcc gcc-c++
```

**CUDA/GPU Issues:**
- The application automatically detects CUDA availability
- Falls back to CPU if CUDA is not available
- For GPU acceleration, ensure CUDA toolkit is installed

### Getting Help

1. Check the logs in `organization_ai.log`
2. Run tests to identify specific issues: `python tests/run_tests.py`
3. Check system requirements and dependencies
4. Verify API keys are correctly configured

## 🎯 Key Features Explained

### AI Model Management
- **Multi-Provider Support**: Seamlessly switch between different AI providers
- **Local and Cloud Models**: Run models locally or use cloud APIs
- **Model Caching**: Efficient caching for better performance
- **Hot-Swapping**: Change models without restarting workflows

### Workflow Builder
- **Visual Interface**: Drag-and-drop workflow creation
- **Agent Configuration**: Configure specialized AI agents for different tasks
- **Dependency Management**: Define execution order and dependencies
- **Real-time Monitoring**: Watch your workflows execute in real-time

### Advanced Features
- **Refinement Loops**: AI agents can iteratively improve their outputs
- **Error Handling**: Robust error handling and recovery mechanisms
- **Progress Tracking**: Detailed progress monitoring and logging
- **Settings Persistence**: Your configurations are saved between sessions

## 📝 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📧 Support

[Add support contact information here]

---

**Built with ❤️ using Python, PyQt6, and cutting-edge AI technologies**
