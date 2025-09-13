#!/usr/bin/env python3
"""
Simple DigitalOcean entry point
"""

from app import app

# Export the Flask app for DigitalOcean
app_do = app
application = app

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
