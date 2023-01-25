import numpy as np  
import cv2
import os
from time import sleep
import kalman
import argparse

def obtain_centers(img_thresh):
    # Find contours
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set the accepted minimum & maximum radius of a detected object
    min_radius_thresh= 3
    max_radius_thresh= 30   

    centers= []
    for c in contours:
        # ref: https://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html
        (x, y), radius = cv2.minEnclosingCircle(c)
        radius = int(radius)

        #Take only the valid circles
        if (radius > min_radius_thresh) and (radius < max_radius_thresh):
            centers.append(np.array([[x], [y]]))
    #cv2.imshow('contours', img_thresh)

    height, width = img_thresh.shape
    min_x, min_y = width, height
    max_x = max_y = 0

    rois = []
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)
        if w > 80 and h > 80:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
            rois.append(cv2.boundingRect(contour))
    return centers, rois



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                                OpenCV. You can process both videos and images.')
    parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='video_cut.mp4')
    parser.add_argument('--substractor', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
    
    args = parser.parse_args()
    filename  = args.input

    if args.substractor == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2()
    else:
        backSub = cv2.createBackgroundSubtractorKNN()

    cap = cv2.VideoCapture(filename)

    KF = kalman.KalmanObject(0.1, 1, 1, 1, 0.1,0.1)
    player1KF = kalman.KalmanObject(0.1, 1, 1, 1, 0.1,0.1)
    player2KF = kalman.KalmanObject(0.1, 1, 1, 1, 0.1,0.1)

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            fgMask = cv2.blur(frame,(5,5))
            fgMask = backSub.apply(fgMask)
            fgMask = cv2.dilate(fgMask, np.ones((5, 5), np.uint8), iterations=5)
            fgMask= cv2.erode(fgMask, np.ones((5, 5), np.uint8), iterations=5)
            ret, fgMask = cv2.threshold(fgMask, 220, 255, cv2.THRESH_BINARY)
            
            # Roudimentary detection
            centers, rois = obtain_centers(fgMask)

            
            # If centroids are detected then track them
            if (len(centers) > 0):

                # Draw the detected circle
                cv2.circle(frame, (int(centers[0][0]), int(centers[0][1])), 10, (0, 191, 255), 2)

                # Predict
                (x, y) = KF.predict()
                # Draw a rectangle as the predicted object position
                cv2.rectangle(frame, (int(x - 15), int(y - 15)), (int(x + 15), int(y + 15)), (255, 0, 0), 2)

                # Update
                (x1, y1) = KF.update(centers[0])

                # Draw a rectangle as the estimated object position
                cv2.rectangle(frame, (int(x1 - 15), int(y1 - 15)), (int(x1 + 15), int(y1 + 15)), (0, 0, 255), 2)

                cv2.putText(frame, "Estimated Position", (int(x1 + 15), int(y1 + 10)), 0, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, "Predicted Position", (int(x + 15), int(y)), 0, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, "Measured Position", (int(centers[0][0] + 15), int(centers[0][1] - 15)), 0, 0.5, (0,191,255), 2)

                # Display the resulting frame
                cv2.imshow('Frame',frame)
                #cv2.imshow('FG Mask', fgMask)

                # Press Q on keyboard to  exit
                if cv2.waitKey(15) & 0xFF == ord('q'):
                    break
        # Break the loop
        else: 
            print("Closing preview")
            break

    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()