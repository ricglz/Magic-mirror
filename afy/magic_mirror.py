'''Magic mirror module'''
from enum import IntEnum
from random import random

from afy.utils import TicToc

class State(IntEnum):
    '''State of the MagicMirror class to know how much to wait'''
    WAIT_LONG = 0
    WAIT_SHORT = 1
    PREDICT = 2

DESIRED_FPS = 24
WAIT_TIMES = [15, 5, 5]

class MagicMirror():
    '''Module that manages the Magic Mirror experience'''
    frames = 0
    seconds = 0

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
        return self.seconds

    def should_predict(self):
        '''Know if there should be a prediction. In addition to changing state'''
        self.frames += 1
        if self.frames == DESIRED_FPS:
            self.frames = 0
            self.seconds += 1
        if self.state == State.PREDICT:
            self._continue_predict()
            return True

        self._wait_time()
        return False

    def reset_tic(self):
        '''Resets timer'''
        self.seconds = 0
        self.frames = 0

    def _update_state(self, new_state: State):
        self.state = new_state
        self.seconds = 0

    def _continue_predict(self):
        if self.toc == self.wait_time:
            self._update_state(State.WAIT_LONG)

    def _wait_time(self):
        if self.toc == self.wait_time:
            prob = random()
            new_state = State.PREDICT if prob <= 0.7 else State.WAIT_SHORT
            self._update_state(new_state)
