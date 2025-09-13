#!/usr/bin/env python3
"""
DigitalOcean App Platform entry point for MatchPoint.ai
This file is specifically created for DigitalOcean deployment compatibility.
"""

import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import the main Flask application
from app import app

# Create the WSGI application object
application = app

# For direct execution (development/testing)
if __name__ == "__main__":
    # Initialize database tables
    with app.app_context():
        from app import db
        db.create_all()
    
    # Run the app
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
