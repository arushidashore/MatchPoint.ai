# matchpoint.ai-v1 (DigitalOcean App Platform - Minimal Version)
import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import logging
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

app = Flask(__name__)

# DigitalOcean App Platform Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'your-jwt-secret-key-change-this')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///matchpoint.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 50 * 1024 * 1024))  # 50MB

# Create directories
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Configure logging for DigitalOcean
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)

# Initialize JWT and SQLAlchemy
jwt = JWTManager(app)
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Define database models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    elbow_angles = db.Column(db.JSON, nullable=False)
    knee_angles = db.Column(db.JSON, nullable=False)
    racket_velocity = db.Column(db.Float, nullable=False)
    feedback_text = db.Column(db.Text, nullable=False)

class Badge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    description = db.Column(db.String(200), nullable=False)

class Player(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    xp = db.Column(db.Integer, default=0)
    badges = db.relationship('Badge', secondary='player_badges', backref='players')

player_badges = db.Table('player_badges',
    db.Column('player_id', db.Integer, db.ForeignKey('player.id'), primary_key=True),
    db.Column('badge_id', db.Integer, db.ForeignKey('badge.id'), primary_key=True)
)

def analyze_swing_minimal(video_path, height, stroke_type):
    """Minimal swing analysis without video processing."""
    try:
        # Generate feedback based on input parameters
        feedback = generate_basic_feedback(height, stroke_type)
        return " ".join(feedback), None
        
    except Exception as e:
        logging.error(f"Error in analyze_swing_minimal: {e}")
        return f"Error processing request: {str(e)}", None

def generate_basic_feedback(height, stroke_type):
    """Generate basic feedback based on input parameters."""
    feedback = []
    
    # Height-based feedback
    try:
        height_float = float(height)
        if height_float < 5.0:
            feedback.append("🟨 Consider adjusting your stance for better balance.")
        elif height_float > 6.5:
            feedback.append("🟩 Good height! Focus on maintaining proper form.")
        else:
            feedback.append("🟩 Your height is well-suited for tennis. Keep practicing!")
    except:
        feedback.append("🟨 Height information could help with personalized feedback.")
    
    # Stroke-specific feedback
    if stroke_type.lower() == 'forehand':
        feedback.append("🟩 Forehand stroke selected. Focus on keeping your elbow at shoulder height and following through.")
    elif stroke_type.lower() == 'backhand':
        feedback.append("🟩 Backhand stroke selected. Maintain good form with proper grip and follow-through.")
    elif stroke_type.lower() == 'serve':
        feedback.append("🟩 Serve selected. Work on your toss consistency and racquet head speed.")
    else:
        feedback.append("🟨 General stroke analysis. Keep practicing with proper form!")
    
    # Add general tips
    feedback.append("💡 Tip: Ensure good lighting and clear background for better analysis.")
    feedback.append("💡 Tip: Record from a side angle to capture full swing motion.")
    feedback.append("💡 Tip: This is a minimal version. Full AI analysis requires additional processing capabilities.")
    
    return feedback

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/ai-platform')
def ai_platform():
    return render_template('index.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    height = request.form.get('height', '5.8')
    stroke_type = request.form.get('stroke_type', 'forehand')
    
    # Validate file size
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning
    
    if file_size > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({'error': f'File too large. Maximum size is {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB.'}), 400
    
    # Save file temporarily
    filename = f"upload_{int(time.time())}_{file.filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    try:
        feedback, video_path = analyze_swing_minimal(file_path, height, stroke_type)
        
        return jsonify({
            'feedback': feedback,
            'video_path': video_path,
            'processing_info': 'DigitalOcean minimal processing',
            'note': 'This is a minimal version for deployment. Full AI analysis requires additional processing capabilities.',
            'file_received': True,
            'file_size_mb': round(file_size / (1024*1024), 2)
        })
    except Exception as e:
        logging.error(f"Analysis error: {e}")
        return jsonify({'error': 'Analysis failed. Please try again.'}), 500
    finally:
        # Clean up uploaded file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logging.error(f"Error cleaning up file: {e}")

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'platform': 'DigitalOcean App Platform',
        'version': 'minimal',
        'features': {
            'file_upload': True,
            'basic_analysis': True,
            'video_processing': False,
            'tensorflow_analysis': False
        }
    })

@app.route('/api/feedback', methods=['POST'])
@jwt_required()
def save_feedback():
    try:
        data = request.get_json()
        feedback_entry = Feedback(
            user_id=data.get('user_id'),
            elbow_angles=data.get('elbow_angles', []),
            knee_angles=data.get('knee_angles', []),
            racket_velocity=data.get('racket_velocity', 0.0),
            feedback_text=data.get('feedback_text', '')
        )
        db.session.add(feedback_entry)
        db.session.commit()
        return jsonify({"message": "Feedback saved successfully!"}), 201
    except Exception as e:
        logging.error(f"Error saving feedback: {e}")
        return jsonify({"error": "Failed to save feedback"}), 500

# Initialize database
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
