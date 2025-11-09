"""
Facial Emotion Detection Web Application
Optimized Flask app for Render deployment
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import sqlite3
import base64
from datetime import datetime
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow import keras

# ======================================================
# 🔧 Flask App Configuration
# ======================================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ======================================================
# 🧠 Load the Trained Model Once (Globally)
# ======================================================
print("🔄 Loading emotion detection model...")
try:
    model = keras.models.load_model('face_emotionModel.h5')
    # Optional: warm up the model to reduce first-request lag
    dummy_input = np.zeros((1, 48, 48, 1))
    model.predict(dummy_input, verbose=0)
    print("✅ Model loaded and warmed up successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# ======================================================
# 😃 Emotion Labels and Responses
# ======================================================
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

EMOTION_RESPONSES = {
    'Angry': "You look angry. What's bothering you? Take a deep breath! 😤",
    'Disgust': "You seem disgusted. Is something not right? 😖",
    'Fear': "You look fearful. Don't worry, everything will be okay! 😨",
    'Happy': "You're smiling! That's wonderful! Keep spreading joy! 😊",
    'Sad': "You look sad. Why are you feeling down? Remember, tough times don't last! 😢",
    'Surprise': "You look surprised! What caught you off guard? 😲",
    'Neutral': "You have a neutral expression. Feeling calm and collected! 😐"
}

# ======================================================
# 🗄️ Database Setup
# ======================================================
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            image_data TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ Database initialized!")

init_db()

# ======================================================
# 🧩 Helper Functions
# ======================================================
def preprocess_image(image):
    """Convert image to grayscale, resize, normalize, reshape"""
    image = image.convert('L').resize((48, 48))
    img_array = np.array(image) / 255.0
    return img_array.reshape(1, 48, 48, 1)

def predict_emotion(image):
    """Predict emotion using the loaded model"""
    if model is None:
        return "Error", 0.0
    try:
        processed = preprocess_image(image)
        predictions = model.predict(processed, verbose=0)
        idx = np.argmax(predictions[0])
        return EMOTIONS[idx], float(predictions[0][idx])
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error", 0.0

def save_to_database(name, email, emotion, confidence, image_data):
    """Insert record into SQLite database"""
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO students (name, email, emotion, confidence, image_data)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, email, emotion, confidence, image_data))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Database error: {e}")
        return False

# ======================================================
# 🌐 Routes
# ======================================================
@app.route('/')
def index():
    return render_template('index.html')

# ✅ Favicon route to prevent 404 in browser console
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form.get('name')
        email = request.form.get('email')
        if not name or not email:
            return jsonify({'success': False, 'error': 'Please fill in all fields!'})

        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded!'})

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected!'})

        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        emotion, confidence = predict_emotion(image)
        if emotion == "Error":
            return jsonify({'success': False, 'error': 'Error processing image. Please try again.'})

        db_success = save_to_database(name, email, emotion, confidence, image_base64)
        if not db_success:
            return jsonify({'success': False, 'error': 'Error saving to database.'})

        message = EMOTION_RESPONSES.get(emotion, "Emotion detected!")
        return jsonify({
            'success': True,
            'emotion': emotion,
            'confidence': round(confidence * 100, 2),
            'message': message
        })
    except Exception as e:
        print(f"Error in predict route: {e}")
        return jsonify({'success': False, 'error': f'An error occurred: {str(e)}'})

@app.route('/stats')
def stats():
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM students')
        total = cursor.fetchone()[0]
        cursor.execute('SELECT emotion, COUNT(*) FROM students GROUP BY emotion ORDER BY COUNT(*) DESC')
        data = cursor.fetchall()
        conn.close()

        html = "<h1>Emotion Detection Stats</h1>"
        html += f"<p><strong>Total Submissions:</strong> {total}</p><ul>"
        for emotion, count in data:
            percent = (count / total * 100) if total > 0 else 0
            html += f"<li>{emotion}: {count} ({percent:.1f}%)</li>"
        html += "</ul><a href='/'>Back to Home</a>"
        return html
    except Exception as e:
        return f"Error: {e}"

# ======================================================
# 🚀 Run App
# ======================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
