import os
import cv2
from mtcnn.mtcnn import MTCNN

# Path to the folder containing images
input_folder_path = "./images"
output_folder_path = "./output"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Create an object of MTCNN detector
detector = MTCNN()

# Function to detect faces in an image, crop them, and save them
def detect_faces_and_save(image, filename):
    faces = detector.detect_faces(image)
    for i, face in enumerate(faces):
        x, y, width, height = face['box']
        cropped_face = image[y:y+height, x:x+width]
        cv2.imwrite(os.path.join(output_folder_path, f"{filename}_face_{i}.jpg"), cropped_face)

# Iterate over the images in the folder
for filename in os.listdir(input_folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the image
        img_path = os.path.join(input_folder_path, filename)
        img = cv2.imread(img_path)

        # Perform face detection, crop faces, and save them
        detect_faces_and_save(img, os.path.splitext(filename)[0])

print("Face detection and saving completed.")
