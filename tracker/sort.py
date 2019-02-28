"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import argparse
import glob
import os.path
import time

import numpy as np
from skimage import io

from filterpy.kalman import KalmanFilter
from numba import jit
from sklearn.utils.linear_assignment_ import linear_assignment


@jit
def iou(bbox_test, bbox_gt):
    """Computes IOU between two bboxes in the form [x1, y1, x2, y2]
    """
    x1, y1 = np.maximum(bbox_test[0:2], bbox_gt[0:2])
    x2, y2 = np.minimum(bbox_test[2:4], bbox_gt[2:4])
    w = np.maximum(0, x2-x1)
    h = np.maximum(0, y2-y1)
    area = lambda x: (x[2]-x[0])*(x[3]-x[1])
    union = w * h
    intersection = area(bbox_test)+area(bbox_gt)-union
    return union / intersection
    
def convert_bbox_to_z(bbox):
    """Convert bbox (x1, y1, x2, y2) to KF.z (x, y, s, r)

        x, y is the center of the box
        s is the scale/ area
        r is the aspect ratio
    """
    x1, y1, x2, y2 = bbox[:4]
    w, h = (x2 - x1), (y2 - y1)
    x, y = (x1 + w/2), (y1 + h/2)
    s, r = (w * h), (w / float(h))
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """Convert KF.x (x, y, s, r) to bbox (x1, y1, x2, y2)

        x1, y1 is the top left
        x2, y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    x1, y1, x2, y2 = (x[0] - w/2), (x[1] - h/2), (x[0] + w/2), (x[1] + h/2)
    if score is None:
        return np.array((x1, y1, x2, y2)).reshape((1, 4))
    else:
        return np.array((x1, y1, x2, y2, score)).reshape(1, 5)

class KalmanBBoxTracker(object):
    count = 0
    def __init__(self, bbox):
        """Init the internel Kalman Filter using bbox

            dim_x = 7, Number of state variables for the Kalman filter
            dim_z = 4, Number of of measurement inputs

            KF.x: init state (x, y, s, r, x', y', s') (dim_x, 1)
                x, y is the bbox center
                s is the bbox area (w * h)
                r is the bbox aspect ratio (w / h)
                x' is the velocity/ variance of x
                y' is the velocity/ variance of y
                s' is the velocity/ variance of s
                update(), predict() will update this variable
            
            KF.F: state transition matrix (dim_x, dim_x)

            KF.H: measurement function (dim_z, dim_x)

            KF.P: covariance matrix (dim_x, dim_x)
                update(), predict() will update this variable
            
            KF.R: measurement noise covariance (dim_z, dim_z)

            KF.Q: process uncertainty  (dim_x, dim_x)
        """
        # define internel kalman filter
        dim_x, dim_z = 7, 4
        self.kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.kf.x = convert_bbox_to_z(bbox)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
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
        self.hits_streak = 0
        self.age = 0    # record the tracker preserved time
        self.objclass = bbox[6]        
    
    def update(self, bbox):
        """Update the state vector with observed bbox"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
    
    def predict(self):
        """Advances the state vector and returns the predicted bounding box estimate

            KF.x: init state (x, y, s, r, x', y', s')
        """
        # area and the area velocity
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        
        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """Returns the current bounding box estimate"""
        return convert_x_to_bbox(self.kf.x)

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """Assigns detections to tracked object with 
        
        Apply Hungarian algorithm by linear_assignment from sklearn
        Returns (matches, unmatched_detections, unmatched_tackers)
    """
    if len(trackers) == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(len(detections)),
                np.empty((0, 5), dtype=int))

    # row: detection, col: trackers
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    matched_indices = linear_assignment(-iou_matrix)

    # records unmatched detection indices
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    
    # records unmatched trackers indices
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    
    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenae(matches, axis=0)
    return (matches,
            np.array(unmatched_detections),
            np.array(unmatched_trackers))

class SORT(object):
    def __init__(self, max_age=1, min_hits=3):
        """Sets key parameter for SORT algorithm"""
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
    
    def update(self, dets):
        """
            this method must be called once for each frame even if no detections
            note: number of objects returned may differ from the the number of detections provides

        Params:
            dets {numpy.ndarray} - in the format [[x1, y1, x2, y2, score], [x1, y1, x2, y2, score], ...]
        """
        self.frame_count += 1

        # get predicted locations from existing tracker
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0] # get the predcit bbox
            trks[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        # row: detection, col: trackers
        # filter and delete invalid detections > appply hungarian algorithm
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(dets, trks)

        # update matched trackers with assigned detections
        for t, trk in enumerate(trks):
            if t not in unmatched_trackers:
                # matched[:, 0] -> trackers, matched[:, 1] -> detections
                # get the matched detection with related tracker
                d = matched[np.where(matched[:, 1] == t)[0], 0]

                # Kalman Filter update function
                trk.update(dets[d, :][0])
        
        # create and initialize new trackers for unmatch detections
        for i in unmatched_detections:
            # dets[i, :] = bbox
            trk = KalmanBBoxTracker(dets[i, :])
            self.trackers.append(trk)
        
        num_trackers = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((
                   d, [trk.id+1], [trk.objclass])).reshape(1, -1))

            num_trackers -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
        


class Sort(object):
  def __init__(self,max_age=1,min_hits=3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0

  def update(self,dets):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.
    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers),5))
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)

    #update matched trackers with assigned detections
    for t,trk in enumerate(self.trackers):
      if(t not in unmatched_trks):
        d = matched[np.where(matched[:,1]==t)[0],0]
        trk.update(dets[d,:][0])

    #create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
          ret.append(np.concatenate((d,[trk.id+1], [trk.objclass])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        #remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))
