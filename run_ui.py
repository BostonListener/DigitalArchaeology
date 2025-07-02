#!/usr/bin/env python3
"""
Archaeological Pipeline Web Interface Launcher

Simple launcher for the archaeological discovery pipeline web interface.

Usage:
    python run_ui.py
    
Then open your browser to: http://localhost:5000

Authors: Archaeological AI Team
License: MIT
"""

import webbrowser
import time
import threading

def open_browser():
    """Open browser to the interface after a short delay."""
    time.sleep(1.5)
    webbrowser.open('http://localhost:5000')

def main():
    """Main launcher function."""
    print("🏺 Archaeological Discovery Pipeline - Web Interface")
    print("=" * 55)
    print("🚀 Starting server...")
    print("📍 Interface: http://localhost:5000")
    print("🔄 Press Ctrl+C to stop")
    
    try:
        from app import app, socketio
        
        # Auto-open browser
        threading.Thread(target=open_browser, daemon=True).start()
        
        # Run the application
        socketio.run(app, debug=False, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

if __name__ == "__main__":
    main()