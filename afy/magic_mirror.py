# -*- coding: utf-8 -*-

from enum import Enum
from random import random

from afy.utils import TicToc

class State(Enum):
    PREDICT = 0
    WAIT_30 = 1
    WAIT_5 = 2

class MagicMirror():
    def __init__(self):
        self.state = State.WAIT_30
        self.tic_toc = TicToc()
        self.tic_toc.tic()

    def should_predict(self):
        ''''''
        if self.state == State.PREDICT:
            self._continue_predict()
            return True

        self._wait_time()
        return False

    def reset_tic(self):
        self.tic_toc.tic()

    def _continue_predict(self):
        if int(self.tic_toc.toc(seconds=True)) == 5:
            self.state = State.WAIT_30
            self.tic_toc.tic()

    def _wait_time(self):
        toc = int(self.tic_toc.toc(seconds=True))
        if (self.state == State.WAIT_30 and toc == 30) or \
           (self.state == State.WAIT_5 and toc == 5):
            prob = random()
            if prob <= 0.7:
                self.state = State.PREDICT
            else:
                self.state = State.WAIT_5
            self.tic_toc.tic()
