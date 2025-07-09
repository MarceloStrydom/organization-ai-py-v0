# Organization AI

A modern desktop application for building and executing intelligent AI workflows using a visual drag-and-drop interface with configurable AI agents and models.

## Features

- 🧠 **Multi-AI Model Support** - OpenAI, Anthropic, Groq, HuggingFace, Ollama
- 🎨 **Visual Workflow Builder** - Drag-and-drop interface for creating AI workflows
- 💻 **Offline Operation** - Complete local model support with caching
- 🔄 **Agent Collaboration** - Multi-agent workflows with refinement loops
- 📊 **Real-time Monitoring** - Live execution tracking and progress updates
- 🎯 **Modern UI** - Professional PyQt6 desktop interface

## Requirements

- Python 3.8+
- PyQt6
- See `requirements.txt` for full dependency list

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd organization-ai-py-v0/organization-ai-py-V0
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python main.py
   ```

## Testing

Run the test suite:
```bash
python run_tests.py
```

## Project Structure

```
organization-ai-py-V0/
├── main.py                    # Application entry point
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── organization_ai.log        # Application logs
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
│   │   ├── top_bar.py       # Top navigation bar
│   │   ├── model_configuration.py # Model configuration dialog
│   │   └── execution_console.py # Execution monitoring
│   └── ui/                   # UI utilities and theming
│       ├── __init__.py
│       └── theme.py         # UI theming and styling
├── data/                     # Application data
│   └── README.md            # Data directory documentation
└── tests/                    # Test suite
    ├── __init__.py
    └── run_tests.py         # Main test runner
```

## License

[Add your license here]
