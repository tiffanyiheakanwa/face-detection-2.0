from flask import Flask, render_template, request
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

# Load pre-trained model
model = load_model("face_emotionModel.h5")

# Define emotion labels (adjust according to your model’s output order)
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Database setup
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

# Function to predict emotion
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
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    email = request.form['email']
    file = request.files['image']

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        emotion = predict_emotion(filepath)
        message = f"You are showing a {emotion.lower()} expression."

        # Customize the message
        if emotion == "Happy":
            message += " Keep smiling!"
        elif emotion == "Sad":
            message = "You are frowning. Why are you sad?"
        elif emotion == "Angry":
            message += " Take a deep breath!"
        elif emotion == "Fear":
            message += " Don't be afraid, it's okay."
        elif emotion == "Surprise":
            message += " You seem surprised!"
        else:
            message += " Stay calm and relaxed."

        # Save user info to database
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (name, email, image_path, emotion) VALUES (?, ?, ?, ?)",
                  (name, email, filepath, emotion))
        conn.commit()
        conn.close()

        return render_template('index.html', message=message, filename=filepath)

    return render_template('index.html', message="Please upload a valid image.")

if __name__ == '__main__':
    app.run(debug=True)
