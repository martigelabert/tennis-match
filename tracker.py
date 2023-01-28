import numpy as np
from kalman import KalmanObject

class Tracker(object):

    self.identifier = 0

    # Here we will store all the kalman filters
    # with relevant data of the 
    self.klm = []

    def __check(roi):


    def __init__(self):
        pass


    def update(sefl, rois):
        for roi in rois:
            # we are gonna check a similar object
            # if there is not we will add it as new
            if not(check(roi)):

        
        return {
            'id': id,
            'previous': bbox,
            'current':  bbox,
            'kM': kalman.KalmanObject(0.1, 1, 1, 1, 0.1, 0.1)
        }