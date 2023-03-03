__author__ = "Keenan Manpearl"
__date__ = "2023/03/01"

"""

reads in an avi video file and returns an arrayy

"""

import cv2
import numpy as np


def read_video(video_path, n_frames=10):
    cap = cv2.VideoCapture(video_path)
    all = []
    i = 0
    while cap.isOpened() and i < n_frames:
        ret, frame = cap.read()
        arr = np.array(frame)
        all.append(arr)
        i += 1
    return np.array(all)
