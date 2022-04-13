from typing import Union
import numpy as np
import cv2 as cv
import os
from tqdm import tqdm


class PreProcess:
    def __init__(self, in_path: str, out_path: str, target_width: int, target_height: int):
        self.in_path = in_path
        self.out_path = out_path
        self.target_width, self.target_height = target_width, target_height

    def compress_imgs(self, rename: bool = True):
        print("\nCompressing...\n")
        if rename:
            fn: Union[int, str] = 0
        for f in tqdm(os.listdir(self.in_path)):
            if f.endswith(".jpg"):
                img = cv.imread(os.path.join(self.in_path, f))
                img = cv.resize(cv.cvtColor(img, cv.COLOR_BGR2GRAY), (self.target_width, self.target_height))
                fn = fn + 1 if rename else f
                cv.imwrite(self.out_path + str(fn) + '.jpg', img)
        print("Compression Complete!")
