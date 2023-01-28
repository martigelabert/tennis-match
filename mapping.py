import sys
import cv2

# Get the path to the video from the command argument
video_path = sys.argv[1]

# Open the video using OpenCV
video = cv2.VideoCapture(video_path)

# Get the first frame of the video
success, frame = video.read()

# Check if we were able to read the frame
if success:
    # Save the frame as an image
    cv2.imwrite("first_frame.jpg", frame)
    print("Saved the first frame of the video to first_frame.jpg")
else:
    print("Unable to read the first frame of the video")

# Release the video capture object
video.release()
