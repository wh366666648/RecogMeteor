import numpy as np
import cv2 as cv
from typing import Dict, Tuple, Optional, Union, List
from collections import deque

from .prep import PreProcess


def preprocess(in_path: str, out_path: str, target_width: int, target_height: int):
    pre = PreProcess(in_path=in_path, out_path=out_path, target_width=int(target_width),
                     target_height=int(target_height))
    pre.compress_imgs()


def blur(img: np.ndarray, kernel_size: Tuple[int, int], sigma_x: float = 0, sigma_y: float = 10, **kwargs):
    return cv.GaussianBlur(img, kernel_size, sigmaX=sigma_x, sigmaY=sigma_y)


class LSD:
    def __init__(self):
        self.__lsd = cv.createLineSegmentDetector(0)
        self.lines = None

    def detect(self, img: np.ndarray, to_return: bool = False) -> Optional[np.ndarray]:
        self.lines = self.__lsd.detect(img)[0]
        if to_return:
            return self.lines

    def drawLines(self, img: np.ndarray):
        if self.lines is None:
            self.detect(img)
        return self.__lsd.drawSegments(img, self.lines)


class HoughTF:
    def __init__(self):
        self.lines = None

    def hough_lines(self, img: np.ndarray, rho: float, theta: float, threshold: Union[float, int]):
        """
        See https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
        :param img: Source image
        :param rho: The resolution of the parameter $\rho$ in pixels
        :param theta: The resolution of the parameter $\theta$ in radian
        :param threshold: Minimum number of intersections to detect a line
        """
        self.lines = cv.HoughLines(image=img, rho=rho, theta=theta, threshold=threshold,
                                   min_theta=0, max_theta=np.pi)

    def drawLines(self, img: np.ndarray):
        if self.lines is not None:
            for i in range(0, len(self.lines)):
                rho = self.lines[i][0][0]
                theta = self.lines[i][0][1]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 200 * (-b)), int(y0 + 200 * a))
                pt2 = (int(x0 - 200 * (-b)), int(y0 - 200 * a))
                print(pt1, pt2)
                cv.line(img, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)


def intensityCheck(img: np.ndarray, lines: np.ndarray, n_samples: int = 10, 
                   threshold: int = 240, check_size: int = 5, criterion: str = 'average'):
    assert lines.shape[1] == 4
    assert np.max(lines[:, [0, 2]]) <= img.shape[1] and np.max(lines[:, [1, 3]]) <= img.shape[0]
    assert isinstance(check_size, int) and check_size % 2 == 1
    criteria = ['mean', 'average', 'maximum', 'minimum', 'median']
    assert criterion in criteria
    gx, gy = np.meshgrid(np.linspace(-(check_size // 2), check_size // 2, check_size), 
                         np.linspace(-(check_size // 2), check_size // 2, check_size), )
    gx, gy = gx.ravel().astype(int), gy.ravel().astype(int)
    pts = np.linspace(0, 1, n_samples).reshape(1, -1)
    for l in lines:
        start = l[0:2].reshape(2, 1)
        end = l[2:].reshape(2, 1)
        mpts = (np.outer(start, 1 - pts) + np.outer(end, pts)).T.astype(int)
        for mpt in mpts:
            px, py = mpt[0] + gx, mpt[1] + gy
        # todo
            
                     

def nearbyImgSearch(lines: Dict, num_of_neighbors: int = 5, tolerance: Tuple[int, int] = (16, 24)):
    imkeys = iter(lines.keys())
    imqueue = deque([next(imkeys)])
    lineQ = deque([lines[imqueue[-1]]])
    while len(imqueue):
        while len(imqueue) < num_of_neighbors:
            imqueue.append(next(imkeys))
            lineQ.append(lines[imqueue[-1]])
        # todo
