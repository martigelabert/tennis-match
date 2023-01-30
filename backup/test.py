import cv2
import numpy as np

# Create a list to store the Kalman filters for each object
kf_list = []

# Read the first frame of the video
ret, frame = video.read()

# Get the bounding box coordinates for all objects in the first frame
bounding_boxes = get_bounding_boxes(frame)

# Create a Kalman filter for each object and add it to the list
for box in bounding_boxes:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
    kf_list.append(kf)

while True:
    # Read the next frame
    ret, frame = video.read()

    # If the video has ended, break the loop
    if not ret:
        break

    # Get the bounding box coordinates for all objects in the current frame
    bounding_boxes = get_bounding_boxes(frame)

    # Loop through all the Kalman filters
    for i, kf in enumerate(kf_list):
        # Get the current bounding box
        box = bounding_boxes[i]

        # Predict the object's location
        predicted_state = kf.predict()

        # Update the Kalman filter with the measured position
        kf.correct(np.array([[np.float32(box[0])], [np.float32(box[1])]]))

        # Draw the bounding box on the frame
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("Tracking", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
video.release()

# Close all windows
cv2.destroyAllWindows()
