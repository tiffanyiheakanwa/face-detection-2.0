"""
Facial Emotion Detection Web Application
Flask app that detects emotions from uploaded face images
"""

from flask import Flask, render_template, request, jsonify
import os
import sqlite3
import base64
from datetime import datetime
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow import keras

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
print("Loading emotion detection model...")
try:
    model = keras.models.load_model('face_emotionModel.h5')
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

# Emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Emotion responses (what to say for each emotion)
EMOTION_RESPONSES = {
    'Angry': "You look angry. What's bothering you? Take a deep breath! 😤",
    'Disgust': "You seem disgusted. Is something not right? 😖",
    'Fear': "You look fearful. Don't worry, everything will be okay! 😨",
    'Happy': "You're smiling! That's wonderful! Keep spreading joy! 😊",
    'Sad': "You look sad. Why are you feeling down? Remember, tough times don't last! 😢",
    'Surprise': "You look surprised! What caught you off guard? 😲",
    'Neutral': "You have a neutral expression. Feeling calm and collected! 😐"
}

# Database setup
def init_db():
    """Initialize the SQLite database"""
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
    print("✓ Database initialized!")

# Initialize database on startup
init_db()

def preprocess_image(image):
    """
    Preprocess the uploaded image for model prediction
    - Converts to grayscale
    - Resizes to 48x48
    - Normalizes pixel values
    """
    # Convert to grayscale
    image = image.convert('L')
    
    # Resize to 48x48 (model input size)
    image = image.resize((48, 48))
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Normalize pixel values to 0-1
    img_array = img_array / 255.0
    
    # Reshape for model input: (1, 48, 48, 1)
    img_array = img_array.reshape(1, 48, 48, 1)
    
    return img_array

def predict_emotion(image):
    """
    Predict emotion from image using the trained model
    Returns: (emotion_label, confidence)
    """
    if model is None:
        return "Error", 0.0
    
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Get the emotion with highest probability
        emotion_index = np.argmax(predictions[0])
        confidence = float(predictions[0][emotion_index])
        emotion = EMOTIONS[emotion_index]
        
        return emotion, confidence
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error", 0.0

def save_to_database(name, email, emotion, confidence, image_data):
    """Save student data and emotion result to database"""
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

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and emotion prediction"""
    try:
        # Get form data
        name = request.form.get('name')
        email = request.form.get('email')
        
        # Validate form data
        if not name or not email:
            return jsonify({
                'success': False,
                'error': 'Please fill in all fields!'
            })
        
        # Get uploaded image
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image uploaded!'
            })
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No image selected!'
            })
        
        # Read and process the image
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert image to base64 for storage
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Predict emotion
        emotion, confidence = predict_emotion(image)
        
        if emotion == "Error":
            return jsonify({
                'success': False,
                'error': 'Error processing image. Please try again.'
            })
        
        # Save to database
        db_success = save_to_database(
            name, email, emotion, confidence, image_base64
        )
        
        if not db_success:
            return jsonify({
                'success': False,
                'error': 'Error saving to database.'
            })
        
        # Get emotion response message
        response_message = EMOTION_RESPONSES.get(emotion, "Emotion detected!")
        
        # Return success response
        return jsonify({
            'success': True,
            'emotion': emotion,
            'confidence': round(confidence * 100, 2),
            'message': response_message
        })
    
    except Exception as e:
        print(f"Error in predict route: {e}")
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        })

@app.route('/stats')
def stats():
    """Display database statistics (optional feature)"""
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        # Get total submissions
        cursor.execute('SELECT COUNT(*) FROM students')
        total = cursor.fetchone()[0]
        
        # Get emotion distribution
        cursor.execute('''
            SELECT emotion, COUNT(*) as count 
            FROM students 
            GROUP BY emotion 
            ORDER BY count DESC
        ''')
        emotion_stats = cursor.fetchall()
        
        conn.close()
        
        stats_html = f"""
        <html>
        <head><title>Statistics</title></head>
        <body style="font-family: Arial; padding: 20px;">
            <h1>Emotion Detection Statistics</h1>
            <p><strong>Total Submissions:</strong> {total}</p>
            <h2>Emotion Distribution:</h2>
            <ul>
        """
        
        for emotion, count in emotion_stats:
            percentage = (count / total * 100) if total > 0 else 0
            stats_html += f"<li>{emotion}: {count} ({percentage:.1f}%)</li>"
        
        stats_html += """
            </ul>
            <a href="/">Back to Home</a>
        </body>
        </html>
        """
        
        return stats_html
    
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    # Run the app
    # Debug=True for development, set to False for production
    app.run(debug=True, host='0.0.0.0', port=5000)