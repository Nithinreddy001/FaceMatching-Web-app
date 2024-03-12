from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # Import CORS
import cv2
import numpy as np
import face_recognition
import csv
import os

app = Flask(__name__)
CORS(app)  # Add this line to enable CORS for your Flask app

# Define the directory where the images are stored
IMAGE_DIRECTORY = r'C:\Users\nithi\Desktop\upwork\v1-bundle\images'

# Load the CSV file and extract stored face encodings
stored_encodings = []
with open('face_encodings.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header row
    for row in reader:
        image_name = row[0]
        face_encoding = np.array(row[1:], dtype=np.float64)
        stored_encodings.append((image_name, face_encoding))

def compare_face_encodings(face_encodings, stored_encodings, tolerance=0.5):
    matched_images = set()
    for face_encoding in face_encodings:
        for stored_encoding in stored_encodings:
            stored_image_name, stored_face_encoding = stored_encoding
            distance = face_recognition.face_distance([stored_face_encoding], face_encoding)
            if distance < tolerance:
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
        
        # Generate URLs for downloading the matched images
        download_urls = [request.host_url + "images/" + image_name for image_name in matched_images]
        
        print("Download URLs:", download_urls)  # Debugging print
        
        return jsonify({'message': 'File successfully uploaded', 'matched_images': download_urls})

@app.route('/images/<path:image_name>', methods=['GET'])
def download_image(image_name):
    image_path = os.path.join(IMAGE_DIRECTORY, image_name)
    if os.path.exists(image_path):
        return send_file(image_path, as_attachment=True)
    else:
        return jsonify({'error': 'Image not found'})

if __name__ == '__main__':
    app.run(debug=True)
