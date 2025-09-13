#!/usr/bin/env python3
"""
WSGI entry point for the MatchPoint.ai application.
This file is used by Gunicorn to serve the Flask application.
"""

import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

from app import app

if __name__ == "__main__":
    app.run()
