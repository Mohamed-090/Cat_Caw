import cv2
import os

def video_to_images(video_path, output_dir, frames_per_second):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frames_per_second)

    success, image = vidcap.read()
    count = 0
    saved_count = 0

    while success:
        if count % frame_interval == 0:
            # Save frame as JPEG file
            frame_filename = os.path.join(output_dir, f"frame{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, image)
            saved_count += 1

        success, image = vidcap.read()
        count += 1

    vidcap.release()
    print(f"Extracted {saved_count} frames at {frames_per_second} fps.")

# Example usage
video_path = "v2.mp4"
output_dir = "save/frames"
frames_per_second = 1  # Extract 2 frames per second

video_to_images(video_path, output_dir, frames_per_second)
