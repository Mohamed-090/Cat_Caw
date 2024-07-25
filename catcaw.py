import cv2
import mediapipe as mp
import math
import numpy as np

import pandas as pd
import joblib  # For saving and loading the model

knn = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def classify_pose(landmarks):
    Lhip_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x
    Lhip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    Lhip_z = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z
    Rhip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x
    Rhip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
    Rhip_z = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z
    Lshoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
    Lshoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    Lshoulder_z = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z
    Rshoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
    Rshoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    Rshoulder_z = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z
    nose_x = landmarks[mp_pose.PoseLandmark.NOSE.value].x
    nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
    nose_z = landmarks[mp_pose.PoseLandmark.NOSE.value].z
    new_data = pd.DataFrame({
    'Left_Hip_X': [Lhip_x],
    'Left_Hip_Y': [Lhip_y],
    'Left_Hip_Z': [Lhip_z],
    'Right_Hip_X': [Rhip_x],
    'Right_Hip_Y': [Rhip_y],
    'Right_Hip_Z': [Rhip_z],
    'Left_Shoulder_X': [Lshoulder_x],
    'Left_Shoulder_Y': [Lshoulder_y],
    'Left_Shoulder_Z': [Lshoulder_z],
    'Right_Shoulder_X': [Rshoulder_x],
    'Right_Shoulder_Y': [Rshoulder_y],
    'Right_Shoulder_Z': [Rshoulder_z],
    'Nose_X': [nose_x],
    'Nose_Y': [nose_y],
    'Nose_Z': [nose_z]
    })
    # Standardize the features
    new_data_scaled = scaler.transform(new_data)

    # Make predictions
    predictions = knn.predict(new_data_scaled)

    # Print the predictions
    #print("Predicted Class:", predictions)
    # Define thresholds for classification
    if predictions[0] == "Class1":
        return "Cat"
    elif predictions[0] == "Class2":
        return "Cow"
    else:
        return "Unknown"

# Load the video
video_path = 'Yoga Seated Cat - Cow Pose.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

# Initialize counters
cat_count = 0
cow_count = 0


pose_class_old = "Cow"
# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Classify the pose
        pose_class = classify_pose(results.pose_landmarks.landmark)
        #print(pose_class, pose_class_old)
        if pose_class == "Cat" and pose_class_old == "Cow":
            pose_class_old = "Cat"
            cat_count += 1
        elif pose_class == "Cow" and pose_class_old == "Cat":
            pose_class_old = "Cow"
            cow_count += 1

        # Draw the pose annotation on the frame
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the classified pose
        cv2.putText(frame, f'Pose: {pose_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Pose: {pose_class}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Pose: {pose_class}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    
    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Press Q on keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

# Print the counts
print(f'Cat Pose Count: {cat_count}')
print(f'Cow Pose Count: {cow_count}')
