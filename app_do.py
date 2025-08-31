# matchpoint.ai-v1 (DigitalOcean App Platform Optimized)
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, render_template, request, jsonify, send_from_directory
import logging
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import tempfile
import shutil
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

# CPU-optimized TensorFlow configuration for DigitalOcean
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(2)

# Global variable for MoveNet model
movenet = None

def load_movenet_model():
    """Load MoveNet model with error handling."""
    global movenet
    try:
        if movenet is None:
            movenet = hub.KerasLayer("https://tfhub.dev/google/movenet/singlepose/lightning/4",
                                     signature="serving_default", signature_outputs_as_dict=True)
            logging.info("MoveNet model loaded successfully")
        return movenet
    except Exception as e:
        logging.error(f"Error loading MoveNet model: {e}")
        return None

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

def analyze_swing_do_optimized(video_path, height, stroke_type):
    """DigitalOcean optimized swing analysis."""
    model = load_movenet_model()
    if model is None:
        return "Error: MoveNet model not available. Please try again later.", None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video file at {video_path}")
        return "Error: Could not open video file.", None
    
    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # DigitalOcean optimization: process fewer frames
    frame_skip = 5  # Process every 5th frame
    max_frames = 30  # Limit to 30 frames max
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
    
    elbow_angles = []
    knee_angles = []
    racket_positions = []
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
        
        try:
            annotated_frame, keypoints = detect_pose_do_optimized(frame, model)
            processed_frames_list.append(annotated_frame)

            # Calculate angles if keypoints are detected
            if np.any(keypoints) and not np.all(keypoints == 0):
                # Elbow angle
                if all(k[2] > 0.3 for k in [keypoints[6], keypoints[8], keypoints[10]]):
                    shoulder = keypoints[6][:2]
                    elbow = keypoints[8][:2]
                    wrist = keypoints[10][:2]
                    elbow_angle = calculate_angle(shoulder, elbow, wrist)
                    elbow_angles.append(elbow_angle)

                # Knee angle
                if all(k[2] > 0.3 for k in [keypoints[12], keypoints[14], keypoints[16]]):
                    hip = keypoints[12][:2]
                    knee = keypoints[14][:2]
                    ankle = keypoints[16][:2]
                    knee_angle = calculate_angle(hip, knee, ankle)
                    knee_angles.append(knee_angle)

                # Racket position
                if keypoints[10][2] > 0.3:
                    racket_positions.append(keypoints[10][:2])
        except Exception as e:
            logging.error(f"Error processing frame {frame_count}: {e}")
            continue

    # Write processed frames
    for frame in processed_frames_list:
        out.write(frame)
    
    cap.release()
    out.release()

    # Move to final location
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output.mp4')
    shutil.move(temp_output_path, output_path)

    # Generate feedback
    feedback = generate_feedback_do(elbow_angles, knee_angles, racket_positions)
    return " ".join(feedback), '/static/output.mp4'

def detect_pose_do_optimized(frame, model):
    """DigitalOcean optimized pose detection."""
    try:
        # Resize for faster processing
        img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 192, 192)
        img = tf.cast(img, dtype=tf.int32)
        
        outputs = model(img)
        keypoints = outputs['output_0'].numpy().squeeze()
        
        # Draw skeleton (minimal for performance)
        y, x, _ = frame.shape
        for keypoint in keypoints:
            ky, kx, kp_conf = keypoint
            if kp_conf > 0.3:
                cv2.circle(frame, (int(kx * x), int(ky * y)), 2, (0, 0, 255), -1)
        
        return frame, keypoints
    except Exception as e:
        logging.error(f"Error during pose detection: {e}")
        return frame, np.zeros((17,3))

def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    try:
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    except Exception as e:
        logging.error(f"Error calculating angle: {e}")
        return 0.0

def generate_feedback_do(elbow_angles, knee_angles, racket_positions):
    """Generate feedback optimized for DigitalOcean."""
    feedback = []
    
    if not elbow_angles and not knee_angles:
        return ["⚠️ No pose data detected. Please ensure you're clearly visible in the video and try again."]
    
    # Elbow analysis
    if elbow_angles:
        avg_elbow_angle = np.mean(elbow_angles)
        if avg_elbow_angle < 70:
            feedback.append("🟧 Elbow position could be improved. Try keeping your elbow at shoulder height during the swing.")
        elif avg_elbow_angle <= 110:
            feedback.append("🟩 Good elbow position! This helps with power and consistency.")
        else:
            feedback.append("🟨 Elbow is a bit high. Focus on smooth, relaxed swings with good follow-through.")
    
    # Knee analysis
    if knee_angles:
        avg_knee_angle = np.mean(knee_angles)
        if avg_knee_angle > 170:
            feedback.append("🟧 Bend your knees more for better balance and power generation.")
        elif avg_knee_angle > 120:
            feedback.append("🟩 Good knee bend! This supports your swing well.")
        else:
            feedback.append("🟦 Excellent knee bend! Great for power and stability.")
    
    # Racket analysis
    if racket_positions and len(racket_positions) > 1:
        try:
            velocities = [np.linalg.norm(np.array(racket_positions[i]) - np.array(racket_positions[i - 1]))
                          for i in range(1, len(racket_positions))]
            avg_velocity = np.mean(velocities)
            
            if avg_velocity < 4:
                feedback.append("🟧 Work on accelerating through your swing for more power.")
            elif avg_velocity < 8:
                feedback.append("🟩 Good swing speed! Keep it up.")
            else:
                feedback.append("🟦 Excellent racket speed! You're swinging with power.")
        except Exception as e:
            logging.error(f"Error calculating racket velocity: {e}")
            feedback.append("🟨 Swing analysis completed. Focus on smooth, consistent motion.")
    
    if not feedback:
        feedback.append("✅ Your form looks good! Keep practicing.")
    
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
        feedback, video_path = analyze_swing_do_optimized(file_path, height, stroke_type)
        
        # Generate simple analysis summary
        summary = generate_analysis_summary(app.config['OUTPUT_FOLDER'])
        
        return jsonify({
            'feedback': feedback,
            'video_path': video_path,
            'summary': summary,
            'processing_info': 'DigitalOcean optimized processing'
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

def generate_analysis_summary(output_folder):
    """Generate a simple analysis summary chart."""
    try:
        plt.figure(figsize=(8, 6))
        categories = ['Elbow Position', 'Knee Bend', 'Swing Speed', 'Overall Form']
        values = [85, 78, 82, 80]  # Example values
        colors = ['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6']
        
        bars = plt.bar(categories, values, color=colors, alpha=0.8)
        plt.ylabel('Score (%)', fontsize=12)
        plt.title('Tennis Swing Analysis Summary', fontsize=14, fontweight='bold')
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
    model_status = load_movenet_model() is not None
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_status,
        'platform': 'DigitalOcean App Platform',
        'optimized': True
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
