# matchpoint.ai-v1 (DigitalOcean App Platform - Simplified Version)
import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
import logging
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import tempfile
import shutil
import time
import json

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

def analyze_swing_simple(video_path, height, stroke_type):
    """Simplified swing analysis without TensorFlow."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Error: Could not open video file at {video_path}")
            return "Error: Could not open video file.", None
        
        # Get video properties
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Process limited frames for performance
        frame_skip = 10  # Process every 10th frame
        max_frames = 20  # Limit to 20 frames max
        processed_frames = 0
        
        # Create temporary output file
        temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_output_path = temp_output.name
        temp_output.close()
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output_path, fourcc, fps//frame_skip, (640, 480))
        except Exception as e:
            logging.error(f"Error creating VideoWriter: {e}")
            cap.release()
            return "Error creating video output.", None
        
        processed_frames_list = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or processed_frames >= max_frames:
                break
                
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
                
            processed_frames += 1
            
            # Resize frame for faster processing
            frame = cv2.resize(frame, (640, 480))
            
            # Add simple text overlay
            cv2.putText(frame, f'Frame {processed_frames}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Stroke: {stroke_type}', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            processed_frames_list.append(frame)
        
        # Write processed frames
        for frame in processed_frames_list:
            out.write(frame)
        
        cap.release()
        out.release()
        
        # Move to final location
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output.mp4')
        shutil.move(temp_output_path, output_path)
        
        # Generate mock feedback based on video properties
        feedback = generate_mock_feedback(frame_width, frame_height, fps, stroke_type)
        return " ".join(feedback), '/static/output.mp4'
        
    except Exception as e:
        logging.error(f"Error in analyze_swing_simple: {e}")
        return f"Error processing video: {str(e)}", None

def generate_mock_feedback(frame_width, frame_height, fps, stroke_type):
    """Generate mock feedback based on video properties."""
    feedback = []
    
    # Analyze video quality
    if frame_width >= 1920 and frame_height >= 1080:
        feedback.append("🟩 High-quality video! This helps with analysis accuracy.")
    elif frame_width >= 1280 and frame_height >= 720:
        feedback.append("🟨 Good video quality. Higher resolution would improve analysis.")
    else:
        feedback.append("🟧 Video quality could be improved. Try recording in HD (720p or higher).")
    
    # Analyze frame rate
    if fps >= 30:
        feedback.append("🟩 Excellent frame rate! This captures smooth motion.")
    elif fps >= 24:
        feedback.append("🟨 Good frame rate. Higher FPS would provide better motion analysis.")
    else:
        feedback.append("🟧 Low frame rate detected. Try recording at 24+ FPS for better analysis.")
    
    # Stroke-specific feedback
    if stroke_type.lower() == 'forehand':
        feedback.append("🟩 Forehand stroke detected. Focus on keeping your elbow at shoulder height and following through.")
    elif stroke_type.lower() == 'backhand':
        feedback.append("🟩 Backhand stroke detected. Maintain good form with proper grip and follow-through.")
    elif stroke_type.lower() == 'serve':
        feedback.append("🟩 Serve detected. Work on your toss consistency and racquet head speed.")
    else:
        feedback.append("🟨 General stroke analysis. Keep practicing with proper form!")
    
    # Add general tips
    feedback.append("💡 Tip: Ensure good lighting and clear background for better analysis.")
    feedback.append("💡 Tip: Record from a side angle to capture full swing motion.")
    
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
    
    # Save file
    filename = f"upload_{int(time.time())}_{file.filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    try:
        feedback, video_path = analyze_swing_simple(file_path, height, stroke_type)
        
        # Generate simple analysis summary
        summary = generate_simple_summary(app.config['OUTPUT_FOLDER'])
        
        return jsonify({
            'feedback': feedback,
            'video_path': video_path,
            'summary': summary,
            'processing_info': 'DigitalOcean simplified processing (TensorFlow not available)',
            'note': 'This is a simplified version for deployment. Full AI analysis requires TensorFlow.'
        })
    except Exception as e:
        logging.error(f"Analysis error: {e}")
        return jsonify({'error': 'Analysis failed. Please try again with a different video.'}), 500
    finally:
        # Clean up uploaded file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logging.error(f"Error cleaning up file: {e}")

def generate_simple_summary(output_folder):
    """Generate a simple analysis summary chart."""
    try:
        plt.figure(figsize=(8, 6))
        categories = ['Video Quality', 'Frame Rate', 'Recording Setup', 'Overall']
        values = [85, 78, 82, 80]  # Example values
        colors = ['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6']
        
        bars = plt.bar(categories, values, color=colors, alpha=0.8)
        plt.ylabel('Score (%)', fontsize=12)
        plt.title('Video Analysis Summary', fontsize=14, fontweight='bold')
        plt.ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        summary_path = os.path.join(output_folder, 'analysis_summary.png')
        plt.savefig(summary_path, dpi=72, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return '/static/analysis_summary.png'
    except Exception as e:
        logging.error(f"Error generating summary: {e}")
        return None

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'platform': 'DigitalOcean App Platform',
        'version': 'simplified',
        'features': {
            'video_processing': True,
            'tensorflow_analysis': False,
            'basic_analysis': True
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
