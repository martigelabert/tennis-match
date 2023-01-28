import numpy as np
import cv2

PLAYER_MODE = 0
BALL_MODE = 1

class Detector(object):
    def __init__(self, mode = PLAYER_MODE, bs = cv2.createBackgroundSubtractorMOG2()):
        self.mode = mode
        self.bs = bs

    def extract(frame):
        if self.mode == BALL_MODE:
            # aproach of color
            pass

        else: # seeking our players
            pass
