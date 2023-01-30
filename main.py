import numpy as np  
import cv2
import os
from time import sleep

import argparse
import random

from scipy.optimize import linear_sum_assignment

class Enitity(object):
    def __init__(self, bbox):
        # Initialization of the Kalman filte
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

        self.current = bbox
        self.anterior = bbox
        self.last_aparience = bbox

        (x,y,w,h) = bbox

        self.time = 0

        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # we will guess that the objects will mantein proportions
        self.h = h
        self.w = w


    def missed(self):
        self.time += 1

    def found(self):
        self.time = 0

    def predicted_state(self):
        return self.kf.predict()

    def predict_bbox(self):
        predicted_state = self.kf.predict()
        x, y = int(predicted_state[0]), int(predicted_state[1])
        return (x, y, self.w, self.h)
    
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

    # original
    fgMask = cv2.blur(fgMask, (8, 8))
    #fgMask = cv2.blur(fgMask, (15, 15))

    fgMask = backSub.apply(fgMask)  # real

    fgMask = cv2.bitwise_and(fgMask, fgMask, mask = cal)


    #origina
    fgMask = cv2.erode(fgMask, np.ones((2, 1), np.uint8), iterations=3)
    fgMask = cv2.dilate(fgMask, np.ones((7, 7), np.uint8), iterations=2)
    
    fgMask = cv2.dilate(fgMask, np.ones((10, 10), np.uint8), iterations=2)
    ret, fgMask = cv2.threshold(fgMask, 150, 200, cv2.THRESH_BINARY)
    #fgMask = cv2.bitwise_and(fgMask, fgMask, mask = cal)

    #ball = fgMask.copy()
    #ball = cv2.bitwise_and(fgMask, ball, mask = b_mask)
    #cv2.imshow('Blue Detector', ball) # to display the blue object output



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
        #cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
        rois.append(cv2.boundingRect(contour))
        
    return rois, fgMask

# Code extracted from
# https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def iou(boxA, boxB):
    (x,y,w,h) = boxA
    boxA = np.array([x,y,x+w,y+h])

    (x,y,w,h) = boxB
    boxB = np.array([x,y,x+w,y+h])

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
    if float(boxAArea + boxBArea - interArea) == 0:
        return -1
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def temporal_coherence(box, rois, ignore=[], frame = [], debug = 0, th = 0.3):
    """Return the index of which the box is more suitable to be"""

    scores = []
    for r in rois:
        scores.append(iou(box, r))
    scores = np.array(scores)
    #print(scores)

    ideal = np.argmax(scores)
    
    #return ideal, rois[np.argmax(scores)]
                    #a = cv2.selectROI(frame)
                    
                    #entities.append(Enitity(box_current))
                    #delete.append() 
    found = False
    i = 0 
    while(not(found)) and i < len(rois):
        
        if ideal in ignore or scores[ideal]<th:
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
        img[:90][:] = 0
        return img


def main():
    entities = []
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


    for _ in range(10):
        # We will ignore the second and third frame
        ret, frame = cap.read()
        _ = get_rois(frame, cal, backSub)


    # Get the bounding box coordinates for all objects in the first frame
    bounding_boxes, _ = get_rois(frame, cal, backSub)

    #mot_tracker = sort.Sort(min_hits=10) 

    # We will start assignating when the ball and all things are on the field
    # Create a Kalman filter for each object and add it to the list
    for box in bounding_boxes:
        entities.append(Enitity(box))

    prev_bbox = []
    # Read until video is completed

    prev = []
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break
        
        #[122,15,89]


        #into_hsv =cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        # changing the color format from BGr to HSV 
        # This will be used to create the mask
        #L_limit=np.array([10,0,10]) # setting the blue lower limit
        #U_limit=np.array([100,100,255]) # setting the blue upper limit
            
    
        #b_mask=cv2.inRange(into_hsv,L_limit,U_limit)
        # creating the mask using inRange() function
        # this will produce an image where the color of the objects
        # falling in the range will turn white and rest will be black
        #blue=cv2.bitwise_and(frame,frame,mask=b_mask)
        # this will give the color to mask.
        #cv2.imshow('Original',frame) # to display the original frame
        #cv2.imshow('Blue Detector',b_mask) # to display the blue object output

        new_bboxes, _ = get_rois(frame, cal, backSub)

        #print(track_bbs_ids.shape)
        #cv2.putText(frame, str(i[4]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

        
        if True:
            matched = []

            clean = frame.copy()
            #print(matched)

            assigned = []
            delete = []

            # assign first players
            for j in range(len(entities)):
                (x,y,w,h) = entities[j].current

                if w > 45 and h > 45:
                    th=0.4
                    # miro si hay alguna bounding box que quiera ser como esta
                    index, box_current = temporal_coherence(entities[j].current, new_bboxes, assigned, th=th)
                    
                    # si no la encuentro mira de usar una prediccion
                    if index== -1:
                        pred_bbox = entities[j].predict_bbox()
                        index, box_current = temporal_coherence(pred_bbox, new_bboxes, assigned, th=th)

                    # si no la encuentro mira de usar la última posicion
                    if index== -1:
                        index, box_current = temporal_coherence(entities[j].anterior, new_bboxes, assigned, th=th)
                    
                    else:
                        matched.append(index)
                        entities[j].found()
                        assigned.append(index)
                        entities[j].update_bbox(box_current)
                        ps = entities[j].predicted_state() # our predict stage
                        pred_bbox = entities[j].predict_bbox()
                        entities[j].correct(box_current)

                        # Draw the bounding box on the frame
                        x, y, w, h = box_current
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                        # Draw the Kalman filter's predicted position on the frame
                        x, y, w, h = pred_bbox
                        #x, y = int(predicted_state[0]), int(predicted_state[1])
                        cv2.rectangle(frame, (x, y), (x+w, y+h), entities[j].color, 2)

            if False:
                copy = clean.copy()
                notmatched = [i for j, i in enumerate(new_bboxes) if j  not in matched]

                if notmatched:
                    print_rois(notmatched, copy)

                    if prev:
                        print_rois(prev, copy)

                    cv2.imshow("notmatched", copy)
                    if cv2.waitKey(0) & 0xFF == ord('f'):
                        pass

            #notmatched = [i for j, i in enumerate(new_bboxes) if j  not in matched]

            #prev = notmatched
            #if len(entities)<3:
            #    for i in range(len(notmatched)):
            #        (x,y,w,h)=notmatched[i]
            #        if w < 45 and h < 45:
            #            entities.append(Enitity(notmatched[i]))


            for j in range(len(entities)):
                (x,y,w,h) = entities[j].current
                if j not in assigned:          


                    th = 0.0
                    # miro si hay alguna bounding box que quiera ser como esta
                    index, box_current = temporal_coherence(entities[j].current, new_bboxes, assigned, th=th)
                    
                    # si no la encuentro mira de usar una prediccion
                    if index== -1:
                        pred_bbox = entities[j].predict_bbox()
                        index, box_current = temporal_coherence(pred_bbox, new_bboxes, assigned, th=th)

                    # si no la encuentro mira de usar la última posicion
                    if index== -1:
                        index, box_current = temporal_coherence(entities[j].anterior, new_bboxes, assigned, th=th)

                    if index == -1:
                        
                        entities[j].missed()
                        if entities[j].time > 10:
                            
                            # if the object has disapeared for more than 5 frames we will delete it
                            delete.append(j)
                            pass
                        else:
                            entities[j].update_bbox(entities[j].predict_bbox())
                            #ps = entities[j].predicted_state() # our predict stage
                            pred_bbox = entities[j].predict_bbox()
                            entities[j].correct(pred_bbox)
                                                    
                            #entities.append(Enitity(box_current))
                            
                    else:
                        matched.append(index)
                        entities[j].found()
                        assigned.append(index)
                        entities[j].update_bbox(box_current)
                        ps = entities[j].predicted_state() # our predict stage
                        pred_bbox = entities[j].predict_bbox()
                        entities[j].correct(box_current)

                        # Draw the bounding box on the frame
                        x, y, w, h = box_current
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                        # Draw the Kalman filter's predicted position on the frame
                        x, y, w, h = pred_bbox
                        #x, y = int(predicted_state[0]), int(predicted_state[1])
                        cv2.rectangle(frame, (x, y), (x+w, y+h), entities[j].color, 2)

        entities = [i for j, i in enumerate(entities) if j not in delete]


        # if we had detections in the previous frame we will check if we have new assignations posible

        #print("prev %i ---- current %i" % (len(prev_bbox), len(new_bboxes)))

        if False:
            copy = clean.copy()
            notmatched = [i for j, i in enumerate(new_bboxes) if j not in matched]

            if notmatched:
                print_rois(notmatched, copy)
                cv2.imshow("notmatched", copy)
                if cv2.waitKey(0) & 0xFF == ord('f'):
                    pass


        if prev_bbox:
            if len(prev_bbox) == len(new_bboxes) and len(matched)!= len(new_bboxes):
                for m in range(len(new_bboxes)):
                    if m not in matched:
                        (x,y,w,h) = new_bboxes[m]
                        if w < 45 and h < 45 and False:
                            copy = clean.copy()
                            print_rois([(new_bboxes[m])], copy)
                            cv2.imshow("Tracking", copy)
                            if cv2.waitKey(0) & 0xFF == ord('f'):
                                pass
                            #cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
                            #rois.append(cv2.boundingRect(contour))
                            pass
                            #entities.append(Enitity(new_bboxes[m]))

        prev_bbox = new_bboxes
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

