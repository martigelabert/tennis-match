import cv2
import numpy as np

video_path = 'vid2.mp4'

# Read the video 
cap = cv2.VideoCapture(video_path)

# Parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas Kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

# Create random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)



while True:
    # Read new frame
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # Calculate Optical Flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        old_gray, frame_gray, p0, None, **lk_params
    )
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
 
    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a[0], b[0]), (c[0], d[0]), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a[0], b[0]), 5, color[i].tolist(), -1)
 
    # Display the demo
    img = cv2.add(frame, mask)
    cv2.imshow("frame", img)
    k = cv2.waitKey(25) & 0xFF
    if k == 27:
        break
 
    # Update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)