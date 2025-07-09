#!/usr/bin/env python3
"""
Organization AI - Setup Script

This script helps users set up the Organization AI application with all
required dependencies and configuration.
"""

import sys
import os
import subprocess
import platform
from pathlib import Path

def print_header():
    """Print setup header."""
    print("=" * 60)
    print("🧠 Organization AI - Setup Script")
    print("=" * 60)
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print("=" * 60)

def check_python_version():
    """Check Python version requirements."""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        print("   Please upgrade Python and try again.")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} is supported")
    return True

def install_system_dependencies():
    """Install system dependencies based on platform."""
    print("\n🔧 Checking system dependencies...")
    
    system = platform.system().lower()
    
    if system == "linux":
        print("📦 Linux detected - checking for required packages...")
        # Check if we can run apt-get (Debian/Ubuntu)
        try:
            result = subprocess.run(["which", "apt-get"], capture_output=True)
            if result.returncode == 0:
                print("Installing OpenGL and PyQt6 dependencies...")
                packages = ["libegl1", "libgl1-mesa-dev", "libxkbcommon0"]
                for package in packages:
                    try:
                        subprocess.run(["sudo", "apt-get", "install", "-y", package], 
                                     check=True, capture_output=True)
                        print(f"✅ Installed {package}")
                    except subprocess.CalledProcessError:
                        print(f"⚠️  Could not install {package} (may already be installed)")
            else:
                print("⚠️  apt-get not found. Please install OpenGL libraries manually if needed.")
        except Exception as e:
            print(f"⚠️  Could not install system dependencies: {e}")
    
    elif system == "darwin":  # macOS
        print("🍎 macOS detected - checking for Homebrew...")
        try:
            subprocess.run(["which", "brew"], check=True, capture_output=True)
            print("Installing Qt6 via Homebrew...")
            subprocess.run(["brew", "install", "qt6"], check=True, capture_output=True)
            print("✅ Qt6 installed via Homebrew")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  Homebrew not found. Qt6 may need to be installed manually.")
    
    elif system == "windows":
        print("🪟 Windows detected - PyQt6 should work out of the box")
        print("✅ No additional system dependencies required")
    
    else:
        print(f"⚠️  Unknown system: {system}. Manual dependency installation may be required.")

def install_python_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing Python dependencies...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    try:
        print("Running: pip install -r requirements.txt")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Python dependencies installed successfully")
            return True
        else:
            print("❌ Failed to install Python dependencies:")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def run_tests():
    """Run application tests."""
    print("\n🧪 Running application tests...")
    
    try:
        result = subprocess.run([
            sys.executable, "tests/run_tests.py"
        ], capture_output=True, text=True, timeout=120)
        
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Some tests failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Tests timed out after 2 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def create_desktop_shortcut():
    """Create desktop shortcut (optional)."""
    print("\n🖥️  Creating desktop shortcut...")
    
    try:
        current_dir = Path(__file__).parent.absolute()
        
        if platform.system().lower() == "linux":
            # Create .desktop file for Linux
            desktop_file = Path.home() / "Desktop" / "OrganizationAI.desktop"
            desktop_content = f"""[Desktop Entry]
Name=Organization AI
Comment=Intelligent AI Workflow Builder
Exec={sys.executable} {current_dir / "main.py"}
Icon={current_dir / "icon.png"}
Terminal=false
Type=Application
Categories=Development;
"""
            desktop_file.write_text(desktop_content)
            os.chmod(desktop_file, 0o755)
            print(f"✅ Desktop shortcut created: {desktop_file}")
            
        elif platform.system().lower() == "windows":
            # Create batch file for Windows
            batch_file = Path.home() / "Desktop" / "Organization AI.bat"
            batch_content = f"""@echo off
cd /d "{current_dir}"
"{sys.executable}" main.py
pause
"""
            batch_file.write_text(batch_content)
            print(f"✅ Desktop shortcut created: {batch_file}")
            
        else:
            print("⚠️  Desktop shortcut creation not supported on this platform")
            
    except Exception as e:
        print(f"⚠️  Could not create desktop shortcut: {e}")

def print_next_steps():
    """Print instructions for next steps."""
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    print("📋 Next Steps:")
    print("1. Run the application:")
    print("   python main.py")
    print("   OR")
    print("   python run.py")
    print()
    print("2. Configure your AI models:")
    print("   - Go to Tools → Model Configuration")
    print("   - Add your API keys for cloud models")
    print("   - Test model connections")
    print()
    print("3. Create your first workflow:")
    print("   - Go to File → New Workflow")
    print("   - Add AI agents and configure them")
    print("   - Build your automation pipeline")
    print()
    print("📖 For more information, see README.md")
    print("🐛 If you encounter issues, check organization_ai.log")
    print("=" * 60)

def main():
    """Main setup function."""
    print_header()
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install system dependencies
    install_system_dependencies()
    
    # Install Python dependencies
    if not install_python_dependencies():
        print("\n❌ Setup failed: Could not install Python dependencies")
        return 1
    
    # Run tests
    if not run_tests():
        print("\n⚠️  Tests failed. Setup completed but there may be issues.")
        print("   Try running the application anyway with: python main.py")
    
    # Optional: Create desktop shortcut
    create_desktop_shortcut()
    
    # Print next steps
    print_next_steps()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())