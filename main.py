import numpy as np  
import cv2
import os
from time import sleep
import argparse
import random

class Enitity(object):
    def __init__(self, bbox, player = 0, is_ball = 0):
        # Initialization of the Kalman filte
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

        self.player = player
        self.hit = 0
        self.is_ball = is_ball

        self.current = bbox
        self.anterior = bbox
        self.last_aparience = bbox

        (x,y,w,h) = bbox

        self.time = 0

        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # we will guess that the objects will mantein proportions
        self.h = h
        self.w = w

    def score_hit(self):
        self.hit  += 1

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


def center(bbox):
    (x,y,w,h) = bbox
    return(x+w//2, y+h//2)


def find_score_player(entities, ball_object, first=0):
    distance = []
    for index,i in enumerate(entities):
        (xe,ye) = center(i.current)
        (xb,yb) = center(ball_object.current)

        #(xe,ye,we,he) = i.current
        #(xb,yb,wb,hb) = ball_object.current

        distance.append(((xe - xb)**2 + (ye - yb)**2)**0.5)
        # if its the first shot we are gona just check which one is  nearest the dude
        if first:
            pass
        else:
            if distance[index] > 150:
                distance[index] = 10000

    return np.argmin(np.array(distance)), distance[np.argmin(np.array(distance))] 

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

    # this is new, seems interesting
    fgMask = cv2.erode(fgMask, np.ones((5,5), np.uint8), iterations=3)
    fgMask = cv2.dilate(fgMask, np.ones((7, 7), np.uint8), iterations=2)

    #fgMask = cv2.erode(fgMask, np.ones((5,7), np.uint8), iterations=3)

    ret, fgMask = cv2.threshold(fgMask, 170, 200, cv2.THRESH_BINARY)
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

    ideal = np.argmax(scores)
    
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
        
        #img[img.shape[0]-250:][:] = 0
        return img

# ahora que tengo a los wachos, los borro de la imagen
def get_rest_of_rois(rois_matched, image_th):

    ball_image = image_th.copy()

    for (x,y,w,h) in rois_matched:
        cv2.rectangle(ball_image, (x,y), (x+w,y+h), (0, 0, 0), -1)

    #fgMask = cv2.erode(ball_image, np.ones((5,5), np.uint8), iterations=3)

    contours, _ = cv2.findContours(ball_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("patata", ball_image)
    r = []
    for contour in contours:
        #if w > 45 and h > 45 or 25 < w < 35 and 25 < h < 35:
        #cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
        r.append(cv2.boundingRect(contour))

    return r, ball_image

def main():

    # The players
    entities = []

    # The ball (entity that will contain the possible ball)
    ball = []

    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction and kalman filters  \
                                                  methods provided by \
                                                  OpenCV. You can process videos and images.')
    parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='video_cut.mp4')
    parser.add_argument('--substractor', type=str, help='Background subtraction method (KNN, MOG2).', default='KNN')
    
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


    for _ in range(5):
        # We will ignore the second and third frame
        ret, frame = cap.read()
        _ = get_rois(frame, cal, backSub)


    # Get the bounding box coordinates for all objects in the first frame
    bounding_boxes, _ = get_rois(frame, cal, backSub)

    #mot_tracker = sort.Sort(min_hits=10) 

    # We will start assignating when the ball and all things are on the field
    # Create a Kalman filter for each object and add it to the list



    for i,box in enumerate(bounding_boxes):
        i+=1
        entities.append(Enitity(box, player=i))

    # las player that hitted the ball
    last = 0
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break
        
        new_bboxes, image_th = get_rois(frame, cal, backSub)

        matched = []
        assigned = []
        delete = []

        clean = frame.copy()

        # assign first players
        for j in range(len(entities)):
            (x,y,w,h) = entities[j].current
            # I will just ignore tiny predictions
            if w > 45 and h > 45:
                th=0.5
                # Search for a bounding box i want to use with the entity we are checking
                index, box_current = temporal_coherence(entities[j].current, new_bboxes, assigned, th=th)
                
                # If we do not found it, we will search using a predicted bounding box
                if index== -1:
                    break
                    pred_bbox = entities[j].predict_bbox()
                    index, box_current = temporal_coherence(pred_bbox, new_bboxes, assigned, th=th)

                # If we still do not find it, we will try to use the las position we have seen it
                if index== -1:
                    index, box_current = temporal_coherence(entities[j].anterior, new_bboxes, assigned, th=th)
                
                else:
                    matched.append(index)
                    entities[j].found()
                    pred_bbox = entities[j].predict_bbox()
                    assigned.append(index)
                    entities[j].update_bbox(box_current)
                    entities[j].correct(box_current)
                    ps = entities[j].predicted_state() # our predict stage
                    #pred_bbox = entities[j].predict_bbox()
                    #entities[j].correct(box_current)

                    # Draw the bounding box on the frame
                    x, y, w, h = box_current
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.circle(frame, center(box_current), 2, (255,255,0), -1)
                    cv2.putText(frame, "Player %i" % entities[j].player , (x, y-10), cv2.FONT_HERSHEY_DUPLEX , 1, 2)

                    # Draw the Kalman filter's predicted position on the frame
                    x, y, w, h = pred_bbox
                    #x, y = int(predicted_state[0]), int(predicted_state[1])
                    cv2.rectangle(frame, (x, y), (x+w, y+h), entities[j].color, 2)

        # these are probably players
        rois_matched = [i for j, i in enumerate(new_bboxes) if j in matched]

        rest, ball_image = get_rest_of_rois(rois_matched, image_th)

        #print_rois(rest, frame=frame)
        #cv2.imshow("balota", ball_image)
        
        if not(ball):
            if rest:
                for i in rest:
                    print(i)
                    ball.append(Enitity(i, is_ball=1))
                # Here we supose that there is only one ball
                # and no artifacts for the first detection
                index, d = find_score_player(entities, ball[0], first=0)
                entities[index].score_hit()
                last = entities[index].player

        else :
            match_ball = []
            ass_ball   = []
            delete_ball= []

            # Assign first players
            for j in range(len(ball)):
                (x,y,w,h) = ball[j].current
                # I will just ignore tiny predictions
                if rest:

                    # We don't need to apply a treshhold here because we are just checking with the bounding box of the ball
                    # and if we have other, we will still checking a good IOU to be assigned.
                    th = 0.0

                    # Search for a bounding box i want to use with the entity we are checking
                    index, box_current = temporal_coherence(ball[j].current, rest, ass_ball, th=th)
                    
                    # If we do not found it, we will search using a predicted bounding box
                    if index== -1:
                        pred_bbox = ball[j].predict_bbox()
                        index, box_current = temporal_coherence(pred_bbox, rest, ass_ball, th=th)

                    # If we still do not find it, we will try to use the las position we have seen it
                    if index== -1:
                        index, box_current = temporal_coherence(ball[j].anterior, rest, ass_ball, th=th)

                    if index == -1:
                        
                        ball[j].missed()
                        if ball[j].time > 25:
                            
                            # if the object has disapeared for more than 5 frames we will delete it
                            delete_ball.append(j)
                            pass
                        else:
                            #entities[j].update_bbox(entities[j].predict_bbox())
                            #ps = entities[j].predicted_state() # our predict stage
                            #pred_bbox = entities[j].predict_bbox()
                            #entities[j].correct(pred_bbox)
                            pass
                            #entities.append(Enitity(box_current))
                    
                    else:
                        #print("aaaaaaaaaaaaaaaaa")
                        # si ha esta más que un poco tiempo y esta muy cerca de un jugador
                        # entonces es que es un hit
                        if True:
                            socore_player,distance = find_score_player(entities, ball[j])
                            if distance >= 1000:
                                pass
                            else:
                                if last == entities[socore_player].player:
                                    pass
                                else:
                                    entities[socore_player].score_hit()
                                    (x1, y1, w, h)=ball[j].current
                                    cv2.putText(frame, "HITAZO!!!!!", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                                    #print("TOCADA")
                                    last = entities[socore_player].player

                        match_ball.append(index)
                        ball[j].found()
                        pred_bbox = ball[j].predict_bbox()
                        ass_ball.append(index)
                        ball[j].update_bbox(box_current)
                        ps = ball[j].predicted_state() # our predict stage
                        ball[j].correct(box_current)

                        # Draw the bounding box on the frame
                        x, y, w, h = box_current
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                        cv2.putText(frame, "Ball", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

                        # draw the kalman filter prediction
                        x, y, w, h = pred_bbox
                        cv2.rectangle(frame, (x, y), (x+w, y+h), ball[j].color, 2)
                else:
                    pass
            # In case of having more than one posible ball, we will get rid of the
            # inconsistent noise
            ball = [i for j, i in enumerate(ball) if j not in delete_ball]

        # Drawing of the Board
        x = 0
        y = 0
        w = 420
        h = 120
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255, 255),-1)
        for i in entities:
            y+=40
            cv2.putText(frame, "Player %i scores -> %i" % (i.player, i.hit), (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, 1, 4)

        cv2.imshow("Tracking", frame)
        entities = [i for j, i in enumerate(entities) if j not in delete]

        # IDEA ; para el video dos meterle un poco de mascara a los catchers
        #if len(entities) > 2:
        #    print("Player overflow")
        
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    print("The Winner is player %i" % last)

    # release video capture
    cap.release()
    # Close
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

