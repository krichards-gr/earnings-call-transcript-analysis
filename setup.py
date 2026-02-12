#!/usr/bin/env python3
"""
Setup Script for Earnings Call Transcript Analysis Tool

This script automates the setup process for the earnings call analysis tool.
It will guide you through installing dependencies, downloading models, and
configuring Google Cloud authentication.
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

def print_step(step_num, total_steps, text):
    print(f"{Colors.OKBLUE}[Step {step_num}/{total_steps}]{Colors.ENDC} {Colors.BOLD}{text}{Colors.ENDC}")

def print_success(text):
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")

def run_command(cmd, description, check=True, capture_output=False):
    """Run a shell command with user feedback"""
    print(f"  Running: {description}...")
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
            return result.stdout.strip()
        else:
            result = subprocess.run(cmd, shell=True, check=check)
            return result.returncode == 0
    except subprocess.CalledProcessError as e:
        if check:
            print_error(f"Failed: {description}")
            print_error(f"Error: {e}")
            return False
        return False

def check_python_version():
    """Verify Python version is 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error(f"Python 3.8+ required. You have Python {version.major}.{version.minor}")
        return False
    print_success(f"Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_command_exists(command):
    """Check if a command exists in PATH"""
    return shutil.which(command) is not None

def install_dependencies():
    """Install Python dependencies"""
    print_step(2, 6, "Installing Python Dependencies")

    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print_error("requirements.txt not found!")
        return False

    # Create virtual environment if it doesn't exist
    if not os.path.exists('.venv'):
        print_info("Creating virtual environment...")
        if not run_command(f"{sys.executable} -m venv .venv", "Create virtual environment"):
            print_error("Failed to create virtual environment")
            return False
        print_success("Virtual environment created")
    else:
        print_info("Virtual environment already exists")

    # Determine the pip command based on OS
    if platform.system() == 'Windows':
        pip_cmd = '.venv\\Scripts\\pip'
        python_cmd = '.venv\\Scripts\\python'
    else:
        pip_cmd = '.venv/bin/pip'
        python_cmd = '.venv/bin/python'

    # Upgrade pip
    print_info("Upgrading pip...")
    run_command(f"{pip_cmd} install --upgrade pip", "Upgrade pip", check=False)

    # Install requirements
    print_info("Installing dependencies from requirements.txt...")
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Install requirements"):
        print_error("Failed to install dependencies")
        return False

    print_success("Dependencies installed successfully")
    return True

def download_spacy_model():
    """Download spaCy language model"""
    print_step(3, 6, "Downloading spaCy Language Model")

    if platform.system() == 'Windows':
        python_cmd = '.venv\\Scripts\\python'
    else:
        python_cmd = '.venv/bin/python'

    print_info("Downloading en_core_web_sm model...")
    if not run_command(f"{python_cmd} -m spacy download en_core_web_sm", "Download spaCy model"):
        print_warning("Failed to download spaCy model automatically")
        print_info("You can download it manually later with: python -m spacy download en_core_web_sm")
        return False

    print_success("spaCy model downloaded")
    return True

def download_ml_models():
    """Download ML models for the analysis pipeline"""
    print_step(4, 6, "Downloading Machine Learning Models")

    if not os.path.exists('download_models.py'):
        print_warning("download_models.py not found. Skipping model download.")
        print_info("Models will be downloaded on first run.")
        return True

    if platform.system() == 'Windows':
        python_cmd = '.venv\\Scripts\\python'
    else:
        python_cmd = '.venv/bin/python'

    print_info("This may take several minutes...")
    if not run_command(f"{python_cmd} download_models.py", "Download ML models"):
        print_warning("Failed to download models")
        print_info("Models will be downloaded automatically on first run")
        return False

    print_success("ML models downloaded")
    return True

def setup_gcloud():
    """Set up Google Cloud authentication"""
    print_step(5, 6, "Configuring Google Cloud Authentication")

    # Check if gcloud is installed
    if not check_command_exists('gcloud'):
        print_warning("Google Cloud SDK (gcloud) not found")
        print_info("To use BigQuery features, install gcloud from:")
        print_info("https://cloud.google.com/sdk/docs/install")

        response = input(f"\n{Colors.BOLD}Do you want to continue without gcloud? (y/n): {Colors.ENDC}").lower()
        if response != 'y':
            return False
        return True

    print_success("gcloud CLI found")

    # Check if already authenticated
    result = run_command('gcloud auth application-default print-access-token',
                        "Check authentication", check=False, capture_output=True)

    if result and len(result) > 0:
        print_success("Already authenticated with Google Cloud")
        return True

    print_info("Setting up Google Cloud authentication...")
    print_info("This will open a browser window for authentication.")

    response = input(f"\n{Colors.BOLD}Authenticate with Google Cloud now? (y/n): {Colors.ENDC}").lower()
    if response == 'y':
        if run_command('gcloud auth application-default login', "Authenticate with Google Cloud"):
            print_success("Authentication successful")
            return True
        else:
            print_error("Authentication failed")
            return False
    else:
        print_info("Skipping authentication. You can authenticate later with:")
        print_info("gcloud auth application-default login")
        return True

def verify_files():
    """Verify required files exist"""
    print_step(6, 6, "Verifying Project Files")

    required_files = [
        'cli_analysis.py',
        'analysis.py',
        'analyzer.py',
        'generate_topics.py',
        'tickers.csv',
        'issue_config_inputs_raw.csv'
    ]

    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print_success(f"Found {file}")
        else:
            print_error(f"Missing {file}")
            missing_files.append(file)

    if missing_files:
        print_error(f"Missing required files: {', '.join(missing_files)}")
        return False

    # Create outputs directory
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        print_success("Created outputs directory")
    else:
        print_info("outputs directory already exists")

    return True

def print_next_steps():
    """Print instructions for next steps"""
    print_header("Setup Complete!")

    print(f"{Colors.OKGREEN}The tool is ready to use!{Colors.ENDC}\n")

    print(f"{Colors.BOLD}Quick Start Commands:{Colors.ENDC}\n")

    if platform.system() == 'Windows':
        activate_cmd = '.venv\\Scripts\\activate'
        python_cmd = '.venv\\Scripts\\python'
    else:
        activate_cmd = 'source .venv/bin/activate'
        python_cmd = '.venv/bin/python'

    print(f"1. Activate virtual environment:")
    print(f"   {Colors.OKCYAN}{activate_cmd}{Colors.ENDC}\n")

    print(f"2. Run a test analysis:")
    print(f"   {Colors.OKCYAN}{python_cmd} cli_analysis.py --test{Colors.ENDC}\n")

    print(f"3. Analyze specific companies:")
    print(f"   {Colors.OKCYAN}{python_cmd} cli_analysis.py --companies AAPL,MSFT,GOOGL{Colors.ENDC}\n")

    print(f"4. Full analysis with date range:")
    print(f"   {Colors.OKCYAN}{python_cmd} cli_analysis.py --start-date 2024-01-01 --end-date 2024-12-31 --mode full{Colors.ENDC}\n")

    print(f"{Colors.BOLD}Documentation:{Colors.ENDC}")
    print(f"   • README.md - Full documentation")
    print(f"   • QUICKSTART.md - Quick start guide")
    print(f"   • {Colors.OKCYAN}{python_cmd} cli_analysis.py --help{Colors.ENDC} - CLI help\n")

    print(f"{Colors.BOLD}Troubleshooting:{Colors.ENDC}")
    print(f"   • If you encounter issues, check that you're using the virtual environment")
    print(f"   • For BigQuery access, ensure you've authenticated with gcloud")
    print(f"   • For models issues, run: {Colors.OKCYAN}{python_cmd} download_models.py{Colors.ENDC}\n")

def main():
    """Main setup workflow"""
    print_header("Earnings Call Transcript Analysis - Setup")

    print(f"{Colors.BOLD}This script will:{Colors.ENDC}")
    print("  • Check Python version")
    print("  • Create virtual environment")
    print("  • Install dependencies")
    print("  • Download spaCy models")
    print("  • Download ML models")
    print("  • Configure Google Cloud authentication")
    print("  • Verify project files\n")

    response = input(f"{Colors.BOLD}Continue with setup? (y/n): {Colors.ENDC}").lower()
    if response != 'y':
        print("Setup cancelled.")
        return

    # Step 1: Check Python version
    print_step(1, 6, "Checking Python Version")
    if not check_python_version():
        sys.exit(1)

    # Step 2: Install dependencies
    if not install_dependencies():
        print_error("Setup failed at dependency installation")
        sys.exit(1)

    # Step 3: Download spaCy model
    download_spacy_model()  # Continue even if this fails

    # Step 4: Download ML models
    download_ml_models()  # Continue even if this fails

    # Step 5: Setup Google Cloud
    setup_gcloud()  # Continue even if this fails

    # Step 6: Verify files
    if not verify_files():
        print_error("Setup failed at file verification")
        sys.exit(1)

    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Setup interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
