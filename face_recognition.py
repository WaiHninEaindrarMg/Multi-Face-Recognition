import os
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import joblib

# Path to the input video file
input_video_path = "./video/face.mp4"
output_folder_path = "./video-output"
output_video_path = "./video-output/output_video.avi"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Load pre-trained VGG16 model
vgg_model = VGG16(weights='imagenet', include_top=False)  # Exclude the top layer for feature extraction

# Load the SVM model from the saved file
svm_model = joblib.load("./classification_model/Linear_SVM_model.pkl")

# Create an object of MTCNN detector
detector = MTCNN()

# Function to extract features using VGG16 model
def extract_features(image):
    img = cv2.resize(image, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    features = vgg_model.predict(img)
    features_flat = features.flatten()  # Flatten the feature tensor
    return features_flat

# Function to detect faces in a frame, classify them, and save them
def detect_faces_and_classify(frame, frame_count, writer):
    faces = detector.detect_faces(frame)
    for i, face in enumerate(faces):
        x, y, width, height = face['box']
        confidence = face['confidence']
        if confidence < 0.4:  # You can adjust the confidence threshold as needed
            continue

        # Compute the area of the bounding box
        bbox_area = width * height

        print(bbox_area)

        if bbox_area < 10000 :
            continue 

        # Extract features for the cropped face using VGG16 model
        features = extract_features(frame[y:y+height, x:x+width])

        # Verify the shape of the extracted features
        if features.shape[0] != 25088:  # Ensure that the number of features matches the SVM model
            print("Invalid number of features")
            continue

        # Predict using the SVM model
        prediction = svm_model.predict([features])[0]

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
        cv2.putText(frame, str(prediction), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Save the annotated frame as an image
    output_image_path = os.path.join(output_folder_path, f"frame_{frame_count}.jpg")
    cv2.imwrite(output_image_path, frame)
    print(f"Saved image: {output_image_path}")

    # Write the frame with annotations to the output video
    writer.write(frame)

# Initialize video capture
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Perform face detection, classify faces, and save them
    detect_faces_and_classify(frame, frame_count, writer)

    # Display the frame or perform additional processing here if needed
    cv2.imshow('Frame', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
cap.release()
writer.release()
cv2.destroyAllWindows()

print("Face detection and classification completed. Video saved.")
