"""
Minimal MatchPoint.ai app for deployment debugging
"""

import os
from flask import Flask, render_template, request, jsonify

# Create Flask app with minimal configuration
app = Flask(__name__)

# Basic configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Create necessary directories
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

@app.route('/')
def landing():
    return """
    <html>
    <head>
        <title>MatchPoint.ai</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .button { background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎾 MatchPoint.ai</h1>
            <p>Welcome to MatchPoint.ai - Your AI-powered tennis swing analyzer!</p>
            <p>Upload your tennis swing video and get instant AI feedback on your technique.</p>
            <a href="/ai-platform" class="button">Start Analysis</a>
            <hr>
            <p><strong>Status:</strong> Minimal app version running successfully!</p>
            <p><a href="/debug">Debug Info</a> | <a href="/test">Test Route</a></p>
        </div>
    </body>
    </html>
    """

@app.route('/ai-platform')
def ai_platform():
    return """
    <html>
    <head>
        <title>MatchPoint.ai - AI Platform</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .form-group { margin: 15px 0; }
            .form-group label { display: block; margin-bottom: 5px; font-weight: bold; }
            .form-group input, .form-group select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
            .button { background: #28a745; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; }
            .button:hover { background: #218838; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎾 MatchPoint.ai - AI Platform</h1>
            <p>Upload your tennis swing video for AI analysis:</p>
            
            <form action="/analyze" method="post" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="video">Select Video File:</label>
                    <input type="file" id="video" name="video" accept="video/*" required>
                </div>
                
                <div class="form-group">
                    <label for="height">Your Height (cm):</label>
                    <input type="number" id="height" name="height" value="170" min="100" max="250" required>
                </div>
                
                <div class="form-group">
                    <label for="stroke_type">Stroke Type:</label>
                    <select id="stroke_type" name="stroke_type" required>
                        <option value="forehand">Forehand</option>
                        <option value="backhand">Backhand</option>
                        <option value="serve">Serve</option>
                    </select>
                </div>
                
                <button type="submit" class="button">Analyze My Swing</button>
            </form>
            
            <hr>
            <p><a href="/">← Back to Home</a></p>
        </div>
    </body>
    </html>
    """

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        height = request.form.get('height', '170')
        stroke_type = request.form.get('stroke_type', 'forehand')
        
        return jsonify({
            'message': 'Video received successfully!',
            'filename': file.filename,
            'height': height,
            'stroke_type': stroke_type,
            'status': 'Analysis feature coming soon - this is the minimal version'
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/debug')
def debug():
    return jsonify({
        'status': 'Minimal app running successfully',
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'output_folder': app.config['OUTPUT_FOLDER'],
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
        'output_folder_exists': os.path.exists(app.config['OUTPUT_FOLDER']),
        'templates_folder': os.path.exists('templates'),
        'static_folder': os.path.exists('static'),
        'flask_version': 'Flask imported successfully'
    })

@app.route('/test')
def test():
    return jsonify({'status': 'Test route working', 'message': 'Minimal app is functional'})

# WSGI application for deployment
application = app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
