"""
Minimal DigitalOcean entry point
"""

# Direct import without any path manipulation
import app

# Export the Flask app
application = app.app

if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 8080))
    app.app.run(host='0.0.0.0', port=port)
