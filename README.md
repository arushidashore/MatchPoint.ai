# MatchPoint.ai

MatchPoint.ai is an AI-powered platform designed to revolutionize the way tennis players analyze and improve their game. By leveraging cutting-edge machine learning models, the platform provides detailed feedback, gamification features, and performance analytics to help players achieve their goals.

## Features

- **AI-Powered Swing Analysis**: Upload your tennis swing videos and receive actionable feedback on your form, including elbow angles, knee angles, and racket velocity.
- **Gamification**: Earn badges, track your XP, and stay motivated with engaging gamification features.
- **Performance Insights**: Access detailed analytics to monitor your improvement and retention over time.
- **Modern Design**: A sleek and user-friendly interface built with Tailwind CSS and JavaScript.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/arushidashore/MatchPoint.ai.git
   ```
2. Navigate to the project directory:
   ```bash
   cd MatchPoint.ai
   ```
3. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the Flask application:
   ```bash
   python app.py
   ```
5. Open your browser and navigate to `http://127.0.0.1:5000` to access the landing page.

## Project Structure

- `app.py`: Core backend logic, including routes and AI model integration.
- `templates/`: HTML templates for the landing page and AI platform.
- `static/`: Static files such as CSS and JavaScript.
- `uploads/`: Directory for uploaded video files.

## Technologies Used

- **Backend**: Flask, Flask-JWT-Extended, Flask-SQLAlchemy
- **Frontend**: Tailwind CSS, JavaScript, Chart.js
- **Machine Learning**: TensorFlow, TensorFlow Hub

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.