__author__ = "Keenan Manpearl"
__date__ = "2023/03/01"

"""

"""
import cv2
import numpy as np
from torch.utils.data import Dataset


class Mitocheck_Dataset(Dataset):
    def __init__(self, image, binary=True):
        super(Mitocheck_Dataset).__init__()
        self.image = image
        self.binary = binary

    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, idx):
        image_seq = np.copy(self.image[idx, :, :])
        image_seq = image_seq.reshape(-1, image_seq.shape[0])
        image_seq = np.transpose(image_seq, (1,0))
        image_list = []
        if self.binary:
            # ## binarize images
            # Otsu algorithm:
            # iteratively searches for the threshold that minimizes
            # the within-class variance, defined as a weighted sum of
            # variances of the two classes (background and foreground)
            for idx in range(9):
                image = image_seq[idx,:]
                nbins = 0.01
                all_colors = image
                total_weight = len(all_colors)
                least_variance = -1
                

                # create an array of all possible threshold values which we want to loop through
                color_thresholds = np.arange(
                    np.min(image) + nbins, np.max(image) - nbins, nbins
                )

                # loop through the thresholds to find the one with the least within class variance
                for color_threshold in color_thresholds:
                    bg_pixels = all_colors[all_colors < color_threshold]
                    weight_bg = len(bg_pixels) / total_weight
                    variance_bg = np.var(bg_pixels)

                    fg_pixels = all_colors[all_colors >= color_threshold]
                    weight_fg = len(fg_pixels) / total_weight
                    variance_fg = np.var(fg_pixels)

                    within_class_variance = (
                        weight_fg * variance_fg + weight_bg * variance_bg
                    )

                    if least_variance == -1 or least_variance > within_class_variance:
                        least_variance = within_class_variance
                image[image >= color_threshold] = 1
                image[image < color_threshold] = 0
                image = image.astype(np.float32)
                image_list.append(image.reshape(-1))

            final_seq = np.array(image_list)
            return final_seq