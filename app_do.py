"""
DigitalOcean App Platform entry point for MatchPoint.ai
"""

import os
import sys
import traceback

# Ensure the current directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print(f"Current directory: {current_dir}")
print(f"Python path: {sys.path[:3]}")

try:
    print("Attempting to import main app...")
    # Import the main Flask application
    from app import app
    application = app
    # Also create app alias for compatibility
    app_do_app = app
    print("✓ Successfully imported Flask app")
    
    # Add a simple route to verify the app is working
    @application.route('/health')
    def health_check():
        return "App is running successfully!"
        
except Exception as e:
    print(f"❌ Import error: {e}")
    print(f"Traceback: {traceback.format_exc()}")
    
    # Fallback: create a simple Flask app with more functionality
    try:
        from flask import Flask, render_template, request, jsonify
        application = Flask(__name__)
        app_do_app = application
        application.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key')
        
        @application.route('/')
        def fallback():
            return """
            <html>
            <head><title>MatchPoint.ai - Deployment Issue</title></head>
            <body>
                <h1>MatchPoint.ai</h1>
                <p>App is running but main module couldn't be imported.</p>
                <p>Error: {}</p>
                <p>This is a fallback version. The main app needs to be fixed.</p>
                <p>Check the deployment logs for more details.</p>
            </body>
            </html>
            """.format(str(e))
        
        @application.route('/health')
        def health():
            return jsonify({"status": "fallback", "error": str(e)})
        
        print("✓ Using fallback Flask app")
        
    except Exception as fallback_error:
        print(f"❌ Even fallback failed: {fallback_error}")
        # Last resort: minimal Flask app
        from flask import Flask
        application = Flask(__name__)
        app_do_app = application
        
        @application.route('/')
        def minimal():
            return f"Minimal app running. Error: {str(e)}"

# Create app attribute for Gunicorn compatibility
app = application

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    application.run(host='0.0.0.0', port=port, debug=False)
