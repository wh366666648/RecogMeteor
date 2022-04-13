import numpy as np
import cv2 as cv
import os
from copy import deepcopy
from matplotlib import pyplot as plt

from mtrlib.dectlib import *

IN_PATH = 'data/original'
OUT_PATH = 'data/resized/'
TARGET_HEIGHT = 800
TARGET_WIDTH = 1200
BLUR_KERNEL = (5, 5)
TO_PREPROCESS = False

if __name__ == '__main__':
    # Problem with Linux-based systems about opencv
    if TO_PREPROCESS:
        preprocess(in_path=IN_PATH, out_path=OUT_PATH, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT)
    """
    try the pipeline:
    - extract the lines with LSD
    - plan B:
        - compare among the images and only keep the differences (help with dealing with the ground?)
        - only compare with nearby images so that the ground movement would not influence a lot
    - plan A: (now doing)
        - Check the line intensity to filter the meteors directly
        - combined with part of plab B: if the lines among the images are similar -> not to check them for intensity
    """
    # lineSet = []
    lsd: LSD = LSD()
    img_lines_pair = {}
    for im in os.listdir(OUT_PATH):
        if im.endswith(".jpg"):
            img = cv.imread(os.path.join(OUT_PATH, im))[:, :, 0]
            img_blurred = blur(img, BLUR_KERNEL)
            # img_copy = deepcopy(img)
            # img_copy = cv.Canny(img_copy, 180, 240)
            lsd.detect(img_blurred)
            detectedLines = lsd.lines.reshape(-1, 4) # lsd.lines [num_of_lines, 1, 4] to [num_of_lines, 4]
            img_lines_pair[im] = detectedLines
            # lineSet.append(lsd.lines.reshape(-1, 4).astype(int))
            # drawlines = lsd.drawLines(img)
            # cv.imshow("Lines", drawlines)
            # _, thrsh = cv.threshold(img, 135, 255, cv.THRESH_TOZERO)
            # _, thrsh = cv.threshold(thrsh, 250, 255, cv.THRESH_TOZERO_INV)
            # cv.imshow("Threshold", thrsh)
            # # lsd.detect(thrsh)
            # # drawlines_thrsh = lsd.drawLines(thrsh)
            # # cv.imshow("Lines and Threshold", drawlines_thrsh)
            # # hough = HoughTF()
            # # hough.hough_lines(img_copy, 1, np.pi / 180, 200)
            # # hough.drawLines(img_copy)
            # # cv.imshow("Hough Transform", img_copy)
    nearbyImgSearch(img_lines_pair, num_of_neighbors=5, tolerance=(TARGET_HEIGHT * 0.02, TARGET_WIDTH * 0.02))
    # todo the naming system: 1 -> 10 -> 11 -> ... -> 2
    print("1")
