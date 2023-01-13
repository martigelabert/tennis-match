import numpy as np  
import cv2
import os
from time import sleep

import argparse


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

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            fgMask = cv2.blur(frame,(5,5))
            fgMask = backSub.apply(fgMask)
            fgMask = cv2.dilate(fgMask, np.ones((5, 5), np.uint8), iterations=3)
            fgMask= cv2.erode(fgMask, np.ones((5, 5), np.uint8), iterations=2)


            # Display the resulting frame
            cv2.imshow('Frame',frame)
            cv2.imshow('FG Mask', fgMask)

            # Press Q on keyboard to  exit
            if cv2.waitKey(15) & 0xFF == ord('q'):
                break
        # Break the loop
        else: 
            print("a")
            break

    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()