import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import argparse

def calibration_mask(vid):
    if vid=='vid2.mp4':
        return cv2.imread('vid2_mask_bin.jpg', cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread('video_cut_mask_bin.jpg', cv2.IMREAD_GRAYSCALE)


def get_rois(frame, cal, backSub):

    fgMask = frame
    #fgMask = cv2.cvtColor(fgMask, cv2.COLOR_BGR2GRAY)
    #fgMask = cv2.equalizeHist(fgMask)
    fgMask = cv2.blur(fgMask, (8, 8))

    fgMask = cv2.bitwise_and(fgMask, fgMask, mask = cal)
    fgMask = backSub.apply(fgMask)  # real

    fgMask = cv2.erode(fgMask, np.ones((2, 1), np.uint8), iterations=3)
    fgMask = cv2.dilate(fgMask, np.ones((7, 7), np.uint8), iterations=2)
    
    fgMask = cv2.dilate(fgMask, np.ones((10, 10), np.uint8), iterations=2)
    ret, fgMask = cv2.threshold(fgMask, 150, 200, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = fgMask.shape
    min_x, min_y = width, height
    max_x = max_y = 0

    rois = []
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)
        #if w > 45 and h > 45 :#or 25 < w < 35 and 25 < h < 35:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
        rois.append(cv2.boundingRect(contour))
        
    return rois

def main():
    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                                OpenCV. You can process both videos and images.')
    parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='video_cut.mp4')
    parser.add_argument('--substractor', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
    #parser.add_argument('--vid', type=str, help='Video selection.', default='video_cut.mp4')
    
    args = parser.parse_args()
    filename  = args.input

    if args.substractor == 'MOG2':
        print("%s selected" % args.substractor)
        backSub = cv2.createBackgroundSubtractorMOG2()
    else:
        print("%s selected" % args.substractor)
        backSub = cv2.createBackgroundSubtractorKNN()

    
    cap = cv2.VideoCapture(filename)

    cal = calibration_mask(filename)



    # Create a list to store the Kalman filters for each object
    kf_list = []

    # Read the first frame of the video
    ret, frame = cap.read()

    # Get the bounding box coordinates for all objects in the first frame
    bounding_boxes = get_rois(frame, cal, backSub)

    # Create a Kalman filter for each object and add it to the list
    for box in bounding_boxes:
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        kf_list.append(kf)

    while True:
        # Read the next frame
        ret, frame = cap.read()

        # If the video has ended, break the loop
        if not ret:
            break

        # Get the bounding box coordinates for all objects in the current frame
        bounding_boxes = get_rois(frame, cal, backSub)

        # Create a list to store the distances between each Kalman filter's predicted position and the current bounding boxes
        distances = []

        # Loop through all the Kalman filters
        for i, kf in enumerate(kf_list):
            # Predict the object's location
            predicted_state = kf.predict()

            # Loop through all the bounding boxes
            for box in bounding_boxes:
                # Calculate the distance between the Kalman filter's predicted position and the current bounding box
                x1, y1, w1, h1 = box
                x2, y2 = int(predicted_state[0]), int(predicted_state[1])
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

                # Add the distance and the current bounding box to the distances list
                distances.append((distance, i, box))

        # Sort the distances list by distance
        distances.sort(key=lambda x: x[0])

        # Create a list to store the Kalman filters that have been associated with a bounding box
        associated_kf = [False for kf in kf_list]

        # Loop through the distances list
        for distance, i, box in distances:
            # If the current Kalman filter has not been associated with a bounding box yet
            if not associated_kf[i]:
                # Update the Kalman filter with the measured position
                kf_list[i].correct(np.array([x1 + w1 / 2, y1 + h1 / 2]))
                associated_kf = [False] * len(kf_list)

                # Loop through the distances list
                for i, (distance, kf_index, box) in enumerate(distances):
                    # If the distance is less than the threshold, associate the Kalman filter with the bounding box
                    if distance < distance_threshold:
                        kf_list[kf_index].correct(np.array([x1 + w1 / 2, y1 + h1 / 2]))
                        associated_kf[kf_index] = True

                # Loop through all the Kalman filters
                for i, kf in enumerate(kf_list):
                    # Predict the next position of the Kalman filter
                    kf.predict()


        # Create a list to store the Kalman filters for new detections
        new_kf_list = []

        # Loop through all the Kalman filters
        for i, kf in enumerate(kf_list):
            # If the current Kalman filter has not been associated with a bounding box
            if not associated_kf[i]:
                # Create a new Kalman filter
                new_kf = cv2.KalmanFilter(4, 2)
                new_kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
                new_kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
                new_kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
                new_kf_list.append(new_kf)

        # Loop through all the bounding boxes
        for box in bounding_boxes:
            # If the current bounding box has not been associated with a Kalman filter
            if box not in [x[2] for x in distances]:
                # Create a new Kalman filter and update it with the measured position
                new_kf = cv2.KalmanFilter(4, 2)
                new_kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
                new_kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
                new_kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
                new_kf.correct(np.array([x1 + w1 / 2, y1 + h1 / 2]))
                new_kf_list.append(new_kf)

        # Replace the old list of Kalman filters with the new one
        kf_list = new_kf_list

        # Draw the bounding boxes and Kalman filter's predicted positions on the frame
        for i, kf in enumerate(kf_list):
            x, y = int(kf.statePost[0]), int(kf.statePost[1])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        for box in bounding_boxes:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Show the frame
        cv2.imshow("Multi-Object Tracking with Kalman Filters", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Clean up
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()

