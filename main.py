import numpy as np  
import cv2
import os
from time import sleep
import kalman
import argparse
import detector

print(detector.BALL_MODE)

valor = 242
identifier = 0

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


def print_rois(rois, frame):
    tmp = frame.copy()
    for (x,y,w,h) in rois:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
    return frame

def generate_entity(bbox, identifier):
    id = identifier
    identifier+=1

    return {
        'id': id,
        'previous': bbox,
        'current':  bbox,
        'kM': kalman.KalmanObject(0.1, 1, 1, 1, 0.1, 0.1)
    }




# Cuando obtenga las rois, las comprobaremos todas con los resultados anteriores, y miraremos cual esta más
# cerca del señor del frame anterior
def obtain_rois(frame, backSub):
    """
        Binarize as much as posible the two players,
        with this method I want the bigger chunks to
        be noticeable, the ball will disapear.

        THIS METHOD WILL NOT DETECT THE BALL,
        THE BALL NEED TO BE FURTHER PROCESED.
    """

    fgMask = cv2.blur(frame, (15, 15))
    fgMask = backSub.apply(fgMask)  # real
    fgMask = cv2.dilate(fgMask, np.ones((3, 3), np.uint8), iterations=5)
    fgMask = cv2.erode(fgMask, np.ones((3, 3), np.uint8), iterations=1)
    #fgMask = cv2.dilate(fgMask, np.ones((3, 3), np.uint8), iterations=3)
    ret, fgMask = cv2.threshold(fgMask, 50, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = fgMask.shape
    min_x, min_y = width, height
    max_x = max_y = 0

    rois = []
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)
        if w > 45 and h > 45:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
            rois.append(cv2.boundingRect(contour))

    # we need the rois
    # return rois
    return rois

def print_a():
    print(valor)

def calibration_mask(vid):

    if vid=='vid2.mp4':
        return cv2.imread('vid2_mask_bin.jpg', cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread('video_cut_mask_bin.jpg', cv2.IMREAD_GRAYSCALE)


def players_det(frame, backSub):
    """
        Binarize as much as posible the two players,
        with this method I want the bigger chunks to
        be noticeable, the ball will disapear.

        THIS METHOD WILL NOT DETECT THE BALL,
        THE BALL NEED TO BE FURTHER PROCESED.
    """
    fgMask = cv2.blur(frame, (15, 15))
    fgMask = backSub.apply(fgMask)  # real
    fgMask = cv2.dilate(fgMask, np.ones((3, 3), np.uint8), iterations=5)
    fgMask = cv2.erode(fgMask, np.ones((3, 3), np.uint8), iterations=1)
    #fgMask = cv2.dilate(fgMask, np.ones((3, 3), np.uint8), iterations=3)
    ret, fgMask = cv2.threshold(fgMask, 50, 255, cv2.THRESH_BINARY)
    return fgMask 


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

    #KF = kalman.KalmanObject(0.1, 1, 1, 1, 0.1,0.1)
    #player1KF = kalman.KalmanObject(0.1, 1, 1, 1, 0.1,0.1)
    #player2KF = kalman.KalmanObject(0.1, 1, 1, 1, 0.1,0.1)

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:

            fgMask = frame
            #fgMask = cv2.cvtColor(fgMask, cv2.COLOR_BGR2GRAY)
            #fgMask = cv2.equalizeHist(fgMask)
            fgMask = cv2.blur(fgMask, (8, 8))

            fgMask = cv2.bitwise_and(fgMask, fgMask, mask = cal)
            fgMask = backSub.apply(fgMask)  # real
            
            fgMask = cv2.dilate(fgMask, np.ones((7, 7), np.uint8), iterations=2)
            fgMask = cv2.erode(fgMask, np.ones((3, 3), np.uint8), iterations=4)
            fgMask = cv2.dilate(fgMask, np.ones((10, 10), np.uint8), iterations=2)
        
            ret, fgMask = cv2.threshold(fgMask, 150, 200, cv2.THRESH_BINARY)
            cv2.imshow('FG', fgMask)

            #fgMask = cv2.erode(fgMask, np.ones((3, 3), np.uint8), iterations=2)
            #fgMask = cv2.dilate(fgMask, np.ones((3, 3), np.uint8), iterations=5)
            #fgMask = cv2.erode(fgMask, np.ones((3, 3), np.uint8), iterations=1)

            contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            height, width = fgMask.shape
            min_x, min_y = width, height
            max_x = max_y = 0

            rois = []
            for contour in contours:
                (x,y,w,h) = cv2.boundingRect(contour)
                min_x, max_x = min(x, min_x), max(x+w, max_x)
                min_y, max_y = min(y, min_y), max(y+h, max_y)
                if w > 45 and h > 25 :#or 25 < w < 35 and 25 < h < 35:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
                    rois.append(cv2.boundingRect(contour))

            fgMask = print_rois(rois=rois, frame=frame)

            # para detectar la pelota puedo mirar cual 

            # Display the resulting frame
            #cv2.imshow('Frame',frame)
            #cv2.imshow('FG Mask', fgMask)

            # Press Q on keyboard to  exit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        # Break the loop
        else: 
            print("Closing preview")
            break

    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

