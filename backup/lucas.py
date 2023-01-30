import numpy as np
import cv2


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
        #cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
        rois.append(cv2.boundingRect(contour))
        
    return rois, fgMask


# Sparse Optical Flow: This method processes the flow vectors of only a few of the most interesting pixels from the entire image, within a frame.

# Dense Optical Flow: Here, the flow vectors of all pixels in the entire frame are processed which makes this technique a little slower

cap = cv2.VideoCapture('video_cut.mp4')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

####################
backSub = cv2.createBackgroundSubtractorKNN()

cal = cv2.imread('video_cut_mask_bin.jpg', cv2.IMREAD_GRAYSCALE)
cal[:90][:] = 0

_, old_gray =get_rois(old_frame, cal, backSub)
#######################
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read()
    #frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, frame_gray = get_rois(old_frame, cal, backSub)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        
        a,b = new.ravel()
        c,d = old.ravel()

        a = int(a)
        b = int(b)
        c=int(c)
        d=int(d)
        
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()