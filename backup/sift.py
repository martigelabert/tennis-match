import numpy as np  
import cv2
import os
from time import sleep
import kalman
import argparse
import detector
import sort

from scipy.optimize import linear_sum_assignment

def get_rois(frame, cal, backSub):

    fgMask = frame

    fgMask = cv2.blur(fgMask, (8, 8))

    fgMask = cv2.bitwise_and(fgMask, fgMask, mask = cal)
    fgMask = backSub.apply(fgMask)  # real

    fgMask = cv2.erode(fgMask, np.ones((2, 1), np.uint8), iterations=3)
    fgMask = cv2.dilate(fgMask, np.ones((7, 7), np.uint8), iterations=2)
    
    fgMask = cv2.dilate(fgMask, np.ones((10, 10), np.uint8), iterations=2)

    #fgMask = cv2.erode(fgMask, np.ones((4, 4), np.uint8), iterations=10)

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
        #if w > 45 and h > 45: #or 25 < w < 35 and 25 < h < 35:
        #cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
        rois.append(cv2.boundingRect(contour))
        
    return rois

# Code extracted from
# https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def iou(box1, box2):
    """
    Implement the intersection over union (IoU) between box1 and box2
        
    Arguments:
        box1 -- first box, numpy array with coordinates (ymin, xmin, ymax, xmax)
        box2 -- second box, numpy array with coordinates (ymin, xmin, ymax, xmax)
    """
    # ymin, xmin, ymax, xmax = box
    
    y11, x11, y21, x21 = box1
    y12, x12, y22, x22 = box2
    
    yi1 = max(y11, y12)
    xi1 = max(x11, x12)
    yi2 = min(y21, y22)
    xi2 = min(x21, x22)
    inter_area = max(((xi2 - xi1) * (yi2 - yi1)), 0)
    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (x21 - x11) * (y21 - y11)
    box2_area = (x22 - x12) * (y22 - y12)
    union_area = box1_area + box2_area - inter_area
    # compute the IoU
    if union_area == 0:
        return -1
    iou = inter_area / union_area
    return iou

def temporal_coherence(box, rois, ignore=[], frame = [], debug = 0):
    """Return the index of which the box is more suitable to be"""

    scores = []
    for r in rois:
        scores.append(iou(box, r))
    scores = np.array(scores)
    #print(scores)

    ideal = np.argmax(scores)
    
    #return ideal, rois[np.argmax(scores)]

    found = False
    i = 0 
    while(not(found)) and i < len(rois):
        if ideal in ignore or scores[ideal]<0.1:
            scores[ideal] = -42 # we will not check it 
            scores
            i+=1

            #if debug:

            #if cv2.waitKey(0) & 0xFF == ord('f'):
            #    break

            ideal = np.argmax(scores)
        else:
            return np.argmax(scores), rois[np.argmax(scores)]
    # return -1 if we dont find our thing
    return -1, box
    
 
def print_rois(rois, frame):
    tmp = frame.copy()
    for (x,y,w,h) in rois:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
    return frame

def calibration_mask(vid):
    if vid=='vid2.mp4':
        return cv2.imread('vid2_mask_bin.jpg', cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread('video_cut_mask_bin.jpg', cv2.IMREAD_GRAYSCALE)
        img[:90][:] = 0
        cv2.imshow("cam", img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            pass
        return img

entities = []

def main():
    first = True
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

    # Read the first frame of the video
    ret, frame = cap.read()
    _ = get_rois(frame, cal, backSub)

    # We will ignore the second and third frame
    ret, frame = cap.read()
    _ = get_rois(frame, cal, backSub)
    ret, frame = cap.read()
    _ = get_rois(frame, cal, backSub)

    ret, frame = cap.read()
    _ = get_rois(frame, cal, backSub)

    ret, frame = cap.read()
    _ = get_rois(frame, cal, backSub)

    # Get the bounding box coordinates for all objects in the first frame
    bounding_boxes = get_rois(frame, cal, backSub)

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break


        rois = get_rois(frame, cal, backSub)
        print_rois(rois, frame=frame )

        #bounding_boxes = new_bounding_boxes
        # Show the frame
        cv2.imshow("Tracking", cv2.resize(frame, dsize=(1820,900), interpolation=cv2.INTER_LINEAR))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

