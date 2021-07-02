'''Module containing custom typings'''
from typing import Sequence, Tuple

import numpy as np

CV2Image = np.ndarray
BBox = Tuple[int, int, int, int]
BBoxes = Sequence[BBox]
