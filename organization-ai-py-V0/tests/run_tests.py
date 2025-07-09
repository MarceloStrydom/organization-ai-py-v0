#!/usr/bin/env python3
"""
Test script for Organization AI application components
"""
import sys
import os

def test_imports():
    """Test all imports"""
    print("🧪 Testing Organization AI imports...")
    
    try:
        # Test PyQt6
        from PyQt6.QtWidgets import QApplication
        print("✅ PyQt6 imported successfully")
        
        # Test core models
        from core.models.data_models import ModelType, ModelConfig, ChatMessage
        print("✅ Core data models imported successfully")
        
        # Test AI integration
        from backend.ai.model_manager import AIModelManager, get_default_models
        from backend.ai.agent_executor import AgentExecutor
        print("✅ AI Integration modules imported successfully")
        
        # Test UI components
        from frontend.components.model_configuration import ModelConfigurationDialog
        from frontend.components.top_bar import TopBar
        from frontend.components.execution_console import ExecutionThread
        print("✅ UI Components imported successfully")
        
        # Test utilities
        from frontend.ui.theme import apply_dark_theme, get_color_scheme
        from core.utils.helpers import setup_logging, ensure_directory
        print("✅ Utilities imported successfully")
        
        # Test configuration
        from config.app_config import get_config
        print("✅ Configuration imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_instantiation():
    """Test component instantiation"""
    print("\n🏗️  Testing component instantiation...")
    
    try:
        from backend.ai.model_manager import AIModelManager, get_default_models
        
        # Test model manager
        model_manager = AIModelManager()
        print("✅ AIModelManager instantiated successfully")
        
        # Test default models
        default_models = get_default_models()
        print(f"✅ Default models loaded: {len(default_models)} models available")
        
        return True
    except Exception as e:
        print(f"❌ Instantiation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration system"""
    print("\n⚙️  Testing configuration system...")
    
    try:
        from config.app_config import get_config
        
        # Test config loading
        config = get_config()
        print("✅ Configuration system loaded successfully")
        
        # Test config access
        theme = config.get('ui.theme', 'dark')
        print(f"✅ Configuration access working: theme = {theme}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_main_app():
    """Test main application initialization without GUI"""
    print("\n🚀 Testing main application initialization...")
    
    try:
        # Import main application
        import main
        print("✅ Main module imported successfully")
        
        # Test without creating GUI (headless mode test)
        print("✅ Application module ready for execution")
        
        return True
    except Exception as e:
        print(f"❌ Main app error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("🧠 ORGANIZATION AI - COMPONENT TEST")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 4
    
    # Run tests
    if test_imports():
        tests_passed += 1
    
    if test_instantiation():
        tests_passed += 1
        
    if test_configuration():
        tests_passed += 1
        
    if test_main_app():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 TEST RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Organization AI is ready to run!")
        print("✅ You can now execute: python main.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 50)
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
