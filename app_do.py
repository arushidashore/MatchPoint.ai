"""
DigitalOcean App Platform entry point for MatchPoint.ai
"""

import os
import sys

# Ensure the current directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    # Import the main Flask application
    from app import app
    application = app
    print("Successfully imported Flask app")
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback: create a simple Flask app
    from flask import Flask
    application = Flask(__name__)
    application.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key')
    
    @application.route('/')
    def fallback():
        return "App is running but main module couldn't be imported"
    
    print("Using fallback Flask app")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    application.run(host='0.0.0.0', port=port, debug=False)
