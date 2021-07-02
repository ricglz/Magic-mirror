'''Module containing custom typings'''
from typing import Sequence, Tuple, Union

import numpy as np

Number = Union[int, float]

CV2Image = np.ndarray
BBox = Union[Tuple[Number, Number, Number, Number], np.ndarray]
BBoxes = Sequence[BBox]
