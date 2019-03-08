from __future__ import print_function

import numpy as np

from filterpy.kalman import KalmanFilter
from sklearn.utils.linear_assignment_ import linear_assignment

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919
}

def convert_bbox_to_z(bbox):
    """Convert bbox (x1, y1, x2, y2) to KF.z (x, y, r, h)
        x, y is the center of bbox
        r is the aspect / ratio
    """
    x1, y1, x2, y2 = bbox[:4]
    x1, y1, x2, y2 = bbox[:4]
    w, h = (x2 - x1), (y2 - y1)
    x, y = (x1 + w/2), (y1 + h/2)
    r = w / float(h)
    return np.array([x, y, r, h]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """Convert Kf.x (x, y, r, h, vx, vy, vr, vh) to bbox (x1, y1, x2, y2)

        x1, y1 is the top-left
        x2, y2 is the bottom-right
    """
    w, h = x[2]*x[3], x[3]
    x1, y1, x2, y2 = (x[0] - w/2), (x[1] - h/2), (x[0] + w/2), (x[1] + h/2)
    if score is None:
        return np.array((x1, y1, x2, y2)).reshape((1, 4))
    else:
        return np.array((x1, y1, x2, y2, score)).reshape(1, 5)


class KalmanBBoxTracker(object):
    count = 0
    def __init__(self, bbox):
        """Simple Kalman filter for tracking bbox in image space

            dim_x = 8, Number of state variables for the Kalman filter
            dim_z = 4, Number of of measurement inputs

            KF.x: init state (x, y, r, h, vx, vy, vr, vh) (dim_x, 1)
                x, y is the bbox center
                r is the bbox aspect ratio (w / h)
                h is the bbox height
                vx, vy, vr, vh is the velocity of x, y, r, h
            
            KF.F: state transition matrix, motion matrix (dim_x, dim_x)

            KF.H: measurement function, update matrix (dim_z, dim_x)

            KF.P: covariance matrix (dim_x, dim_x)
                update(), predict() will update this variable
            
            KF.R: measurement noise covariance (dim_z, dim_z)

            KF.Q: process uncertainty  (dim_x, dim_x)
        """
        dim_x, dim_z = 8, 4
        self.kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.kf.F = np.array([[1,0,0,0,1,0,0,0],
                              [0,1,0,0,0,1,0,0],
                              [0,0,1,0,0,0,1,0],
                              [0,0,0,1,0,0,0,1],
                              [0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        self.kf.P[4:, 4:] *= 1000.  # set unobservable initial velocities with high uncertainty
        self.kf.P *= 10.
        self.kf.R[2:, 2:] *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.id = KalmanBBoxTracker.count
        KalmanBBoxTracker.count += 1

        self.time_since_update = 0
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0    # record the tracker preserved time
        self.objclass = bbox[6]
