import cv2
import mediapipe as mp
import pandas as pd
import os

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(image):
    """Extract landmarks from an image."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        # Extract landmarks for left and right hip and shoulder positions
        def get_landmark(landmark_type):
            landmark = landmarks[landmark_type.value] if landmarks[landmark_type.value] else None
            return (landmark.x if landmark else None,
                    landmark.y if landmark else None,
                    landmark.z if landmark else None)

        left_hip_x, left_hip_y, left_hip_z = get_landmark(mp_pose.PoseLandmark.LEFT_HIP)
        right_hip_x, right_hip_y, right_hip_z = get_landmark(mp_pose.PoseLandmark.RIGHT_HIP)
        left_shoulder_x, left_shoulder_y, left_shoulder_z = get_landmark(mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder_x, right_shoulder_y, right_shoulder_z = get_landmark(mp_pose.PoseLandmark.RIGHT_SHOULDER)
        nose_x, nose_y, nose_z = get_landmark(mp_pose.PoseLandmark.NOSE)

        return {
            'Left_Hip_X': left_hip_x,
            'Left_Hip_Y': left_hip_y,
            'Left_Hip_Z': left_hip_z,
            'Right_Hip_X': right_hip_x,
            'Right_Hip_Y': right_hip_y,
            'Right_Hip_Z': right_hip_z,
            'Left_Shoulder_X': left_shoulder_x,
            'Left_Shoulder_Y': left_shoulder_y,
            'Left_Shoulder_Z': left_shoulder_z,
            'Right_Shoulder_X': right_shoulder_x,
            'Right_Shoulder_Y': right_shoulder_y,
            'Right_Shoulder_Z': right_shoulder_z,
            'Nose_X': nose_x,
            'Nose_Y': nose_y,
            'Nose_Z': nose_z
        }
    else:
        return {
            'Left_Hip_X': None,
            'Left_Hip_Y': None,
            'Left_Hip_Z': None,
            'Right_Hip_X': None,
            'Right_Hip_Y': None,
            'Right_Hip_Z': None,
            'Left_Shoulder_X': None,
            'Left_Shoulder_Y': None,
            'Left_Shoulder_Z': None,
            'Right_Shoulder_X': None,
            'Right_Shoulder_Y': None,
            'Right_Shoulder_Z': None,
            'Nose_X': None,
            'Nose_Y': None,
            'Nose_Z': None
        }

def process_folder(folder_path, class_name):
    """Process all images in a folder and extract landmarks."""
    data = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            landmarks = extract_landmarks(image)
            data.append({'Class': class_name, **landmarks})

    return data

def save_to_excel(data, output_file):
    """Save the extracted data to an Excel file."""
    df = pd.DataFrame(data)
    df.to_excel(output_file, index=False)

# Example usage
folder_class1 = 'output_images/cat'
folder_class2 = 'output_images/cow'
output_file = 'landmarks_data.xlsx'

class1_data = process_folder(folder_class1, 'Class1')
class2_data = process_folder(folder_class2, 'Class2')
all_data = class1_data + class2_data

save_to_excel(all_data, output_file)

print(f"Data saved to {output_file}")
