import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Image:
    img: np.ndarray
    width: Optional[int] = 0
    height: Optional[int] = 0
