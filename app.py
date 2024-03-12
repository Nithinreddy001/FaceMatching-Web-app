from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import cv2
import numpy as np
import face_recognition
import csv

app = Flask(__name__)
CORS(app)  # Add this line to enable CORS for your Flask app

# Load the CSV file and extract stored face encodings
stored_encodings = []
with open('face_encodings.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header row
    for row in reader:
        image_name = row[0]
        face_encoding = np.array(row[1:], dtype=np.float64)
        stored_encodings.append((image_name, face_encoding))

def compare_face_encodings(face_encodings, stored_encodings, tolerance=0.6):
    matched_images = set()
    for face_encoding in face_encodings:
        for stored_encoding in stored_encodings:
            stored_image_name, stored_face_encoding = stored_encoding
            distance = face_recognition.face_distance([stored_face_encoding], face_encoding)
            if distance <= tolerance:
                matched_images.add(stored_image_name)
    return matched_images

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        img = face_recognition.load_image_file(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(img)
        if len(face_locations) == 0:
            return jsonify({'message': 'No faces found in the uploaded image', 'matched_images': []})

        face_encodings = face_recognition.face_encodings(img, face_locations)

        matched_images = compare_face_encodings(face_encodings, stored_encodings)
        
        return jsonify({'message': 'File successfully uploaded', 'matched_images': list(matched_images)})

if __name__ == '__main__':
    app.run(debug=True)
