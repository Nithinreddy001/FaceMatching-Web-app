import os
import cv2
import numpy as np
import face_recognition
import csv

# Directory containing images
directory = r'C:\Users\nithi\Desktop\upwork\v1-bundle\images'

# Check if face_encodings.csv exists and load existing filenames
existing_filenames = set()
if os.path.exists('face_encodings.csv') and os.path.getsize('face_encodings.csv') > 0:
    with open('face_encodings.csv', mode='r') as file:
        reader = csv.reader(file)
        try:
            next(reader)  # Skip header
            for row in reader:
                existing_filenames.add(row[0])
        except StopIteration:
            pass

# Create or open CSV file
with open('face_encodings.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    if file.tell() == 0:
        writer.writerow(["Image Name", "Face Encoding"])

    # Iterate over all image files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.jpeg') or filename.endswith('.jpg') or filename.endswith('.png'):
            if filename in existing_filenames:
                print(f"{filename} already exists in the CSV file.")
                continue
            
            image_path = os.path.join(directory, filename)
            
            # Load image
            img = face_recognition.load_image_file(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get face encodings
            face_encodings = face_recognition.face_encodings(img)

            if len(face_encodings) == 0:
                print(f"No face found in {filename}")
                continue

            # Write face encodings to CSV
            for face_encoding in face_encodings:
                writer.writerow([filename, *face_encoding])

            # No need to display the image here
