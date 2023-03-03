__author__ = "Keenan Manpearl"
__date__ = "2023/03/01"

"""
Converts movies into a pickle file for input to td-vae model 
"""
import cv2
import numpy as np
import pathlib
import pickle
from video_reader import read_video

movie_dir = pathlib.Path("/home/keenanmanpearl/Desktop/mitocheck_movies/movies")
save_path = pathlib.Path("mitocheck.pkl")
n_frames = 10

mitocheck = []
for plate_dir in movie_dir.iterdir():
    for well_dir in plate_dir.iterdir():
        # processed videos end in .avi
        for movie_path in well_dir.glob("*.avi"):
            vid = read_video(str(movie_path))
            # dim 1 = frame
            # dim 2 = height
            # dim 3 = width
            # dim 4 = channels, we do not need this
            vid = vid[:, :, :, 0]
            mitocheck.append(vid)
mitocheck_arr = np.array(mitocheck)

output = open(save_path, "wb")
pickle.dump(mitocheck_arr, output)
output.close()
