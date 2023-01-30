import numpy as np
import cv2
import argparse
from scipy.optimize import linear_sum_assignment

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
            #cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
        rois.append(cv2.boundingRect(contour))
        
    return rois

class MultiTracker:
    def __init__(self):
        self.trackers = []
        self.track_ids = []
        self.next_track_id = 1

    def update(self, bounding_boxes):
        # Create new Kalman filters for any new bounding boxes
        for i, box in enumerate(bounding_boxes):
            if i not in self.track_ids:
                # Create a new Kalman filter for the new bounding box
                tracker = cv2.KalmanFilter(4, 2)
                x, y, w, h = box
                tracker.statePost = np.array([x+w/2, y+h/2, 0, 0], dtype=np.float32)
                self.trackers.append(tracker)
                self.track_ids.append(i)

    def predict(self):
        # Predict the next location of each object using the Kalman filters
        for tracker in self.trackers:
            tracker.predict()

    def correct(self, bounding_boxes):
        # Correct the Kalman filters with the measured bounding boxes
        new_track_ids = [-1]*len(bounding_boxes)

        # Use the Hungarian algorithm to match the measured bounding boxes to the closest Kalman filter
        if len(self.trackers) > 0:
            cost = np.zeros((len(self.trackers), len(bounding_boxes)))
            for i, tracker in enumerate(self.trackers):
                for j, box in enumerate(bounding_boxes):
                    x, y, w, h = box
                    predicted = (int(tracker.statePost[0]), int(tracker.statePost[1]))
                    actual = (x + w//2, y + h//2)
                    cost[i][j] = ((predicted[0] - actual[0]) ** 2 + (predicted[1] - actual[1]) ** 2)**0.5
            row_ind, col_ind = linear_sum_assignment(cost)
            for i, j in zip(row_ind, col_ind):
                new_track_ids[j] = self.track_ids[i]
                self.trackers[i].correct(np.array([bounding_boxes[j][0] + bounding_boxes[j][2] // 2,
                                                  bounding_boxes[j][1] + bounding_boxes[j][3] // 2], dtype=np.float32))

        # Remove any Kalman filters for which there were no measured bounding boxes
        for i in reversed(range(len(self.trackers))):
            if self.track_ids[i] not in new_track_ids:
                self.trackers.pop(i)
                self.track_ids.pop(i)

        # Assign new track IDs to any newly detected bounding boxes
        for i, box_id in enumerate(new_track_ids):
            if box_id == -1:
                new_track_ids[i] = self.next_track_id
                self.next_track_id += 1
        self.track_ids = new_track_ids

    def print_predictions(self, frame):
        """Return the predicted bounding boxes"""
        boxes = []
        for tracker in self.trackers:
            # Predict the next state of the object
            tracker.predict()
            # Get the predicted position and size
            x, y = int(tracker.statePre[0]), int(tracker.statePre[1])
            w, h = int(tracker.statePre[2]), int(tracker.statePre[3])
            # Calculate the bounding box coordinates
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        #return frame
        
    def draw_boxes(self, frame):
        # Draw the bounding boxes and track IDs on the frame
        for i, tracker in enumerate(self.trackers):
            x, y = int(tracker.statePost[0]), int(tracker.statePost[1])
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(frame, str(self.track_ids[i]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


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

# Initialize the multi-object tracker
mt = MultiTracker()

# Process video frames
while True:
    # Capture the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    bounding_boxes = get_rois(frame, cal, backSub)

    # Update the multi-object tracker with the new bounding boxes
    mt.update(bounding_boxes)

    # Predict the next location of each object
    mt.predict()

    # get predicted bounding boxes
    #predicted_boxes = mt.get_predicted_boxes()

    #mt.print_predictions(frame)

    # Correct the Kalman filters with the measured bounding boxes
    mt.correct(bounding_boxes)

    # Draw the bounding boxes and track IDs on the frame
    mt.draw_boxes(frame)

    # Display the frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




