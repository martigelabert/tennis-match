import numpy as np  
import cv2
import os
from time import sleep
import kalman
import argparse
import detector
import sort

from scipy.optimize import linear_sum_assignment

class Enitity(self):
    def __init__(self, bbox):
        # Initialization of the Kalman filte
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

        self.current = bbox
        self.anterior = bbox
        (x,y,w,h) = bbox

        # we will guess that the objects will mantein proportions
        self.h = h
        self.w = w

    def predicted_state(self):
        return self.kf.predict()
    
    def correct(self, box):
        self.kf.correct(np.array([[np.float32(box[0])], [np.float32(box[1])]]))

    def update_bbox(self, bbox):
        self.anterior = self.current
        self.current = bbox
        
        (x,y,w,h) = bbox

        # we will guess that the objects will mantein proportions
        self.h = h
        self.w = w



def get_rois(frame, cal, backSub):

    fgMask = frame

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
        #if w > 45 and h > 45 or 25 < w < 35 and 25 < h < 35:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
        rois.append(cv2.boundingRect(contour))
        
    return rois

# Code extracted from
# https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def temporal_coherence(box, rois, ignore=[]):
    """Return the index of which the box is more suitable to be"""
    score = []
    for r in rois:
        scores.append(iou(box, r))
    scores = np.array(scores)

    ideal = np.argmax(scores)
    
    found = False
    i = 0 
    while(not(found)) and i < len(rois):
        if ideal in ignore or scores[ideal]<0.3:
            scores[ideal] = -42 # we will not check it 
            scores
            i+=1
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
        img[:60][:] = 0
        cv2.imshow("cam", img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        return img



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

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        new_bounding_boxes = get_rois(frame, cal, backSub)

        # Use Hungarian algorithm to match the bounding boxes between frames
        cost_matrix = np.zeros((len(bounding_boxes), len(new_bounding_boxes)))
        for i in range(len(bounding_boxes)):
            for j in range(len(new_bounding_boxes)):
                cost_matrix[i, j] = np.linalg.norm(np.array(bounding_boxes[i]) - np.array(new_bounding_boxes[j]))

        # just getting the distance
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Update the Kalman filter for each matched object
        for i, j in zip(row_ind, col_ind):
            kf = kf_list[i]
            box = new_bounding_boxes[j]
            predicted_state = kf.predict()
            kf.correct(np.array([[np.float32(box[0])], [np.float32(box[1])]]))

            # Draw the bounding box on the frame
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Draw the Kalman filter's predicted position on the frame
            x, y = int(predicted_state[0]), int(predicted_state[1])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

        #bounding_boxes = new_bounding_boxes
        # Show the frame
        cv2.imshow("Tracking", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()