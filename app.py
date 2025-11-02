from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
import sqlite3

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your trained model
model = load_model("face_emotionModel.h5")

# Define emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT,
                image_path TEXT,
                emotion TEXT
                )''')
    conn.commit()
    conn.close()

init_db()

# Emotion prediction function
def predict_emotion(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img.reshape(1, 48, 48, 1) / 255.0
    pred = model.predict(img)
    emotion = emotions[np.argmax(pred)]
    return emotion

# Routes
@app.route('/')
def index():
    return render_template('index.html')  # Make sure index.html exists in a /templates folder

@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form['name']
        email = request.form['email']
        file = request.files['image']

        if not file:
            return render_template('index.html', message="Please upload an image.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict emotion
        emotion = predict_emotion(filepath)
        message = f"You are showing a {emotion.lower()} expression."

        # Add emotion-based message
        responses = {
            "Happy": "Keep smiling! 😊",
            "Sad": "You look sad. Hope you feel better soon. 💙",
            "Angry": "Take a deep breath — it’ll be okay. 😤",
            "Fear": "Don't be afraid, you’re safe. 🤗",
            "Surprise": "Wow, you seem surprised! 😲",
            "Disgust": "Something seems off. 😕",
            "Neutral": "Calm and collected — nice! 😌"
        }
        message += " " + responses.get(emotion, "")

        # Save data to database
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (name, email, image_path, emotion) VALUES (?, ?, ?, ?)",
                  (name, email, filepath, emotion))
        conn.commit()
        conn.close()

        return render_template('index.html', message=message, filename=filepath)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Required for Render
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
