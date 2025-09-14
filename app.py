import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, render_template, request, jsonify, send_from_directory
import logging  # Import logging
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = Flask(__name__)

# Production configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static'  # For processed video
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Configure logging for production
if app.debug:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

# Load MoveNet model lazily to avoid import-time failures
movenet = None

def get_movenet_model():
    """Load MoveNet model on first use to avoid import-time failures"""
    global movenet
    if movenet is None:
        try:
            movenet = hub.KerasLayer("https://tfhub.dev/google/movenet/singlepose/lightning/4",
                                   signature="serving_default", signature_outputs_as_dict=True)
            logging.info("MoveNet model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load MoveNet model: {e}")
            raise e
    return movenet

# Placeholder for stroke classification model
# stroke_classifier = hub.KerasLayer("https://tfhub.dev/google/vision-transformer/small/1", 
#                                    signature="serving_default", signature_outputs_as_dict=True)

# Initialize JWT and SQLAlchemy
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
jwt = JWTManager(app)
db = SQLAlchemy(app)

# Initialize Flask-Migrate
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

# Add gamification models
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

def analyze_swing(video_path, height, stroke_type):
    """Analyzes the tennis swing and generates feedback."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video file at {video_path}")
        return "Error: Could not open video file.", None
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output.mp4')
    try:
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'H264'), fps, (frame_width, frame_height))
    except Exception as e:
        logging.error(f"Error creating VideoWriter: {e}")
        cap.release()
        return "Error creating video. Check codec and file permissions.", None
    elbow_angles = []
    knee_angles = []
    racket_positions = []
    all_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, keypoints = detect_pose(frame)
        all_frames.append(annotated_frame)

        # Calculate elbow angle (using right arm)
        shoulder = keypoints[6][:2]
        elbow = keypoints[8][:2]
        wrist = keypoints[10][:2]
        if all(k[2] > 0.3 for k in [keypoints[6], keypoints[8], keypoints[10]]):
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            elbow_angles.append(elbow_angle)

        # Calculate knee angle (using right leg)
        hip = keypoints[12][:2]
        knee = keypoints[14][:2]
        ankle = keypoints[16][:2]
        if all(k[2] > 0.3 for k in [keypoints[12], keypoints[14], keypoints[16]]):
            knee_angle = calculate_angle(hip, knee, ankle)
            knee_angles.append(knee_angle)

        # Track racket position
        racket_position = keypoints[10][:2]
        if keypoints[10][2] > 0.3:
            racket_positions.append(racket_position)

    # Write all frames at once.
    for frame in all_frames:
        out.write(frame)
    cap.release()
    out.release()

    # Analyze the angles
    avg_elbow_angle = np.mean(elbow_angles) if elbow_angles else 0
    avg_knee_angle = np.mean(knee_angles) if knee_angles else 0

    feedback = []
    if avg_elbow_angle < 50:
        feedback.append(
            "🟥 Your elbow is extremely low, severely limiting power and increasing injury risk. "
            "Focus on drills to keep your elbow raised at shoulder height during the swing."
        )
    elif avg_elbow_angle < 70:
        feedback.append(
            "🟧 Elbow is low, restricting swing speed and control. Keep practicing raising your elbow during your swings."
        )
    elif avg_elbow_angle < 90:
        feedback.append(
            "🟨 Elbow position is slightly low. Raising it will help increase power and consistency."
        )
    elif avg_elbow_angle <= 110:
        feedback.append(
            "🟩 Great elbow position! This helps generate power and consistent contact."
        )
    else:
        feedback.append(
            "🟨 Your elbow is a bit too high, which can cause late contact or overextension. "
            "Work on smooth, relaxed swings with good follow-through."
        )

    if avg_knee_angle > 180:
        feedback.append(
            "🟥 Your knees are fully locked, which kills balance and power. Bend more before swinging."
        )
    elif avg_knee_angle > 170:
        feedback.append(
            "🟧 Knees too straight, limiting balance and power generation. Try to lower your center of gravity."
        )
    elif avg_knee_angle > 150:
        feedback.append(
            "🟨 Knee bend is okay but could be improved. Lower your stance slightly for better stability."
        )
    elif avg_knee_angle > 120:
        feedback.append(
            "🟩 Good knee bend, supporting power and balance."
        )
    else:
        feedback.append(
            "🟦 Excellent knee bend! This maximizes power and stability during your swings."
        )

    if racket_positions and len(racket_positions) > 1:
        velocities = [np.linalg.norm(np.array(racket_positions[i]) - np.array(racket_positions[i - 1]))
                      for i in range(1, len(racket_positions))]
        avg_velocity = np.mean(velocities)

        if avg_velocity < 2:
            feedback.append(
                "🟥 Your swing speed is very slow. Focus on accelerating through contact using your hips and shoulders."
            )
        elif avg_velocity < 4:
            feedback.append(
                "🟧 Swing speed is low; work on driving your swing through the ball to generate more power."
            )
        elif avg_velocity < 6:
            feedback.append(
                "🟨 Decent swing speed but room to increase power and explosiveness."
            )
        elif avg_velocity < 8:
            feedback.append(
                "🟩 Good swing speed! You're generating solid racket acceleration."
            )
        else:
            feedback.append(
                "🟦 Excellent racket speed — you're swinging with power and efficiency."
            )

        path_variation = np.std([pos[1] for pos in racket_positions])

        if path_variation > 20:
            feedback.append(
                "🟥 Your racket path is very unstable, leading to inconsistent shots. Practice keeping a smooth and linear swing."
            )
        elif path_variation > 15:
            feedback.append(
                "🟧 Your racket path is unstable. Focus on controlling your swing arc for better contact."
            )
        elif path_variation > 10:
            feedback.append(
                "🟨 Your swing path is somewhat stable but can be smoother."
            )
        elif path_variation > 5:
            feedback.append(
                "🟩 Good stable racket path, which supports consistent hitting."
            )
        else:
            feedback.append(
                "🟦 Excellent racket path control! Your swing is smooth and repeatable."
            )
    else:
        feedback.append("⚠️ Not enough data to analyze racket velocity and path.")

    if not feedback:
        feedback.append("✅ Your form looks good! Keep practicing.")


    return " ".join(feedback), '/static/output.mp4'

def detect_pose(frame, movenet_model=None):
    """Detects pose and draws the skeleton on the frame."""
    if movenet_model is None:
        movenet_model = get_movenet_model()
    
    img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 192, 192)
    img = tf.cast(img, dtype=tf.int32)
    try:
        outputs = movenet_model(img)
        keypoints = outputs['output_0'].numpy().squeeze()
    except Exception as e:
        logging.error(f"Error during MoveNet inference: {e}")
        return frame, np.zeros((17,3)) # Return frame and zeroed keypoints

    y, x, _ = frame.shape
    for keypoint in keypoints:
        ky, kx, kp_conf = keypoint
        if kp_conf > 0.3:
            cv2.circle(frame, (int(kx * x), int(ky * y)), 5, (0, 0, 255), -1)

    # Draw lines between keypoints
    connections = [(0, 1), (0, 2), (1, 3), (2, 4),  # Head to shoulders and arms
                   (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # Body and legs
                   (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)]  # Feet
    for connection in connections:
        start_point = keypoints[connection[0]]
        end_point = keypoints[connection[1]]
        if start_point[2] > 0.3 and end_point[2] > 0.3:
            cv2.line(frame,
                     (int(start_point[1] * x), int(start_point[0] * y)),
                     (int(end_point[1] * x), int(end_point[0] * y)),
                     (0, 0, 255), 2)

    # Track the racket (assuming it's the wrist point)
    racket_point = keypoints[10]  # Assuming wrist is the racket point
    if racket_point[2] > 0.3:
        cv2.circle(frame, (int(racket_point[1] * x), int(racket_point[0] * y)), 5, (255, 0, 0), -1)

    return frame, keypoints

def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def classify_stroke(video_path):
    """Classifies the stroke type from the video."""
    logging.warning("Stroke classification is temporarily disabled.")
    return "Stroke classification disabled"

@app.route('/')
def landing():
    try:
        return render_template('landing.html')
    except Exception as e:
        logging.error(f"Error rendering landing.html: {e}")
        return f"""
        <html>
        <head><title>MatchPoint.ai</title></head>
        <body>
            <h1>Welcome to MatchPoint.ai</h1>
            <p>Tennis swing analysis using AI</p>
            <p><a href="/ai-platform">Go to AI Platform</a></p>
            <p>Error: {str(e)}</p>
        </body>
        </html>
        """, 500

@app.route('/ai-platform')
def ai_platform():
    try:
        return render_template('index.html')
    except Exception as e:
        logging.error(f"Error rendering index.html: {e}")
        return f"""
        <html>
        <head><title>MatchPoint.ai - AI Platform</title></head>
        <body>
            <h1>MatchPoint.ai - AI Platform</h1>
            <p>Upload your tennis swing video for analysis</p>
            <form action="/analyze" method="post" enctype="multipart/form-data">
                <input type="file" name="video" accept="video/*" required><br><br>
                <input type="number" name="height" placeholder="Your height (cm)" required><br><br>
                <select name="stroke_type" required>
                    <option value="forehand">Forehand</option>
                    <option value="backhand">Backhand</option>
                    <option value="serve">Serve</option>
                </select><br><br>
                <button type="submit">Analyze Swing</button>
            </form>
            <p>Error: {str(e)}</p>
        </body>
        </html>
        """, 500

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
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        feedback, video_path = analyze_swing(file_path, height, stroke_type)
        os.remove(file_path)  # Clean up uploaded file
        
        return jsonify({'feedback': feedback, 'video_path': video_path})
        
    except Exception as e:
        logging.error(f"Error in analyze route: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    logging.debug(f"Serving static file: {filename}") # Log the requested filename
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

@app.route('/feedback', methods=['POST'])
@jwt_required()
def feedback():
    user_id = request.json.get('user_id')
    elbow_angles = request.json.get('elbow_angles')
    knee_angles = request.json.get('knee_angles')
    racket_velocity = request.json.get('racket_velocity')
    feedback_text = request.json.get('feedback_text')

    feedback_entry = Feedback(
        user_id=user_id,
        elbow_angles=elbow_angles,
        knee_angles=knee_angles,
        racket_velocity=racket_velocity,
        feedback_text=feedback_text
    )
    db.session.add(feedback_entry)
    db.session.commit()

    return jsonify({"message": "Feedback saved successfully!"}, 201)

@app.route('/history', methods=['GET'])
@jwt_required()
def history():
    user_id = request.args.get('user_id')
    feedback_entries = Feedback.query.filter_by(user_id=user_id).all()
    return jsonify([{
        "elbow_angles": entry.elbow_angles,
        "knee_angles": entry.knee_angles,
        "racket_velocity": entry.racket_velocity,
        "feedback_text": entry.feedback_text
    } for entry in feedback_entries])

@app.route('/gamification', methods=['GET'])
@jwt_required()
def gamification():
    user_id = request.args.get('user_id')
    player = Player.query.filter_by(id=user_id).first()
    if not player:
        return jsonify({'error': 'Player not found'}), 404

    badges = [{'name': badge.name, 'description': badge.description} for badge in player.badges]
    return jsonify({'xp': player.xp, 'badges': badges})

@app.route('/add-xp', methods=['POST'])
@jwt_required()
def add_xp():
    user_id = request.json.get('user_id')
    xp = request.json.get('xp')
    player = Player.query.filter_by(id=user_id).first()
    if not player:
        return jsonify({'error': 'Player not found'}), 404

    player.xp += xp
    db.session.commit()
    return jsonify({'message': 'XP added successfully', 'new_xp': player.xp})

# Debug route to help identify issues
@app.route('/debug')
def debug():
    try:
        info = {
            'status': 'App is running',
            'python_version': sys.version,
            'flask_version': Flask.__version__,
            'tensorflow_available': 'tensorflow' in sys.modules,
            'opencv_available': 'cv2' in sys.modules,
            'numpy_available': 'numpy' in sys.modules,
            'upload_folder': app.config['UPLOAD_FOLDER'],
            'output_folder': app.config['OUTPUT_FOLDER'],
            'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
            'output_folder_exists': os.path.exists(app.config['OUTPUT_FOLDER']),
            'templates_folder': os.path.exists('templates'),
            'static_folder': os.path.exists('static'),
            'landing_template': os.path.exists('templates/landing.html'),
            'index_template': os.path.exists('templates/index.html')
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'Debug failed'}), 500

# WSGI application for deployment
application = app

if __name__ == '__main__':
    # Initialize database tables
    with app.app_context():
        db.create_all()
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
