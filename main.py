"""
Little Medical Reader - Main Entry Point
Launches the V2 Streamlit application for processing medical PDF documents.

Author: GitHub Copilot
Date: 2025-07-14
"""

import subprocess
import sys
from pathlib import Path
import logging
import socket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def find_available_port(start_port=8501, max_attempts=10):
    """
    Find an available port starting from the given port.
    
    Args:
        start_port: Port to start checking from
        max_attempts: Maximum number of ports to check
        
    Returns:
        Available port number or None if no port found
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(('localhost', port))
                return port
        except OSError:
            continue
    return None

def main():
    """
    Main entry point for Little Medical Reader.
    Launches the V2 Streamlit application.
    """
    print("🏥 Welcome to Little Medical Reader V2!")
    print("📄 Advanced PDF processing for medical journal articles")
    print("-" * 50)
    
    # Get the path to the V2 app
    v2_app_path = Path(__file__).parent / "V2" / "main.py"
    
    if not v2_app_path.exists():
        print(f"❌ Error: V2 app not found at {v2_app_path}")
        logger.error(f"V2 app not found at {v2_app_path}")
        sys.exit(1)
    
    try:
        print("🚀 Launching Little Medical Reader V2...")
        logger.info("Starting Streamlit app from main.py")
        
        # Find an available port
        available_port = find_available_port()
        if not available_port:
            print("❌ Error: Could not find an available port (tried 8501-8510)")
            logger.error("No available ports found")
            sys.exit(1)
        
        if available_port != 8501:
            print(f"ℹ️  Port 8501 is busy, using port {available_port} instead")
            logger.info(f"Using alternative port: {available_port}")
        
        # Launch the Streamlit app
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(v2_app_path),
            "--server.address", "localhost",
            "--server.port", str(available_port),
            "--browser.gatherUsageStats", "false"
        ]
        
        print(f"🌐 App will be available at: http://localhost:{available_port}")
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching the app: {e}")
        logger.error(f"Error launching Streamlit app: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Thanks for using Little Medical Reader!")
        logger.info("Application stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
