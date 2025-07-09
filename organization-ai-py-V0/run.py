#!/usr/bin/env python3
"""
Organization AI - Application Launcher

This script provides an easy way to launch the Organization AI application
with proper environment setup and error handling.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version meets requirements."""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required.")
        print(f"   Current version: {sys.version}")
        return False
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        ("PyQt6", "PyQt6"),
        ("torch", "torch"), 
        ("transformers", "transformers"),
        ("aiohttp", "aiohttp"),
        ("requests", "requests")
    ]
    missing = []
    
    for display_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(display_name)
    
    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def setup_environment():
    """Setup environment variables and paths."""
    # Add current directory to Python path
    current_dir = Path(__file__).parent.absolute()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # Set QT platform for better compatibility
    if "QT_QPA_PLATFORM" not in os.environ:
        # Use default platform (don't force offscreen unless needed)
        pass

def run_tests():
    """Run application tests before starting."""
    print("🧪 Running application tests...")
    try:
        result = subprocess.run([sys.executable, "tests/run_tests.py"], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Some tests failed:")
            print(result.stdout)
            if result.stderr:
                print("Errors:")
                print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Tests timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def launch_application():
    """Launch the main application."""
    print("🚀 Starting Organization AI...")
    try:
        # Import and run the main application
        import main
        return main.main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure all dependencies are installed.")
        return 1
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return 1

def main():
    """Main launcher function."""
    print("=" * 50)
    print("🧠 Organization AI - Application Launcher")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Run tests (optional, can be skipped with --skip-tests)
    if "--skip-tests" not in sys.argv:
        if not run_tests():
            print("\n⚠️  Tests failed. Continue anyway? (y/N): ", end="")
            if input().lower() != 'y':
                print("Exiting...")
                return 1
            print()
    
    # Launch application
    exit_code = launch_application()
    
    if exit_code == 0:
        print("\n👋 Organization AI closed successfully.")
    else:
        print(f"\n❌ Application exited with code: {exit_code}")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())