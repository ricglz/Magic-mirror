# -*- coding: utf-8 -*-

from enum import Enum
from random import random

from afy.utils import TicToc

class State(Enum):
    '''State of the MagicMirror class to know how much to wait'''
    WAIT_LONG = 0
    WAIT_SHORT = 1
    PREDICT = 2

WAIT_TIMES = [15, 5, 5]

class MagicMirror():
    '''Module that manages the Magic Mirror experience'''
    def __init__(self):
        self.state = State.WAIT_LONG
        self.tic_toc = TicToc()
        self.tic_toc.tic()

    @property
    def wait_time(self):
        '''Based on the current state returns how much time to wait'''
        return WAIT_TIMES[int(self.state)]

    @property
    def toc(self):
        '''Get amount of seconds that is waiting since the timer started'''
        return int(self.tic_toc.toc(seconds=True))

    def should_predict(self):
        '''Know if there should be a prediction. In addition to changing state'''
        if self.state == State.PREDICT:
            self._continue_predict()
            return True

        self._wait_time()
        return False

    def reset_tic(self):
        '''Resets timer'''
        self.tic_toc.tic()

    def _continue_predict(self):
        if self.toc == self.wait_time:
            self.state = State.WAIT_LONG
            self.tic_toc.tic()

    def _wait_time(self):
        if self.toc == self.wait_time:
            prob = random()
            if prob <= 0.7:
                self.state = State.PREDICT
            else:
                self.state = State.WAIT_SHORT
            self.tic_toc.tic()
