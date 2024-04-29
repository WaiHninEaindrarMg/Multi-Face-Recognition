import os
import cv2
from mtcnn.mtcnn import MTCNN

# Path to the input video file
input_video_path = "./video/face.mp4"
output_folder_path = "./video-output"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Create an object of MTCNN detector
detector = MTCNN()

# Function to detect faces in a frame, crop them, and save them
def detect_faces_and_save(frame, frame_count):
    faces = detector.detect_faces(frame)
    for i, face in enumerate(faces):
        x, y, width, height = face['box']
        cropped_face = frame[y:y+height, x:x+width]
        cv2.imwrite(os.path.join(output_folder_path, f"frame_{frame_count}_face_{i}.jpg"), cropped_face)

# Initialize video capture
cap = cv2.VideoCapture(input_video_path)

frame_count = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Perform face detection, crop faces, and save them
    detect_faces_and_save(frame, frame_count)

    # Display the frame or perform additional processing here if needed
    cv2.imshow('Frame', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

print("Face detection and saving completed.")
