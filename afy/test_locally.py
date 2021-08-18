'''Module to test using a video the local predictor'''
from os import makedirs
from random import seed

import cv2
from tqdm import tqdm

from afy.local_arguments import local_opt as opt
from afy.utils import get_predictor
from afy.helper_functions import load_images, prepare_image

def create_objects():
    '''Create necessary objects for the prediction'''
    predictor = get_predictor(opt, opt.fsgan)
    avatars, _ = load_images(opt)
    predictor.set_source_image(avatars[opt.desired_avatar])
    return predictor, avatars

def frame_iter(capture):
    '''Iterator to get the frame'''
    def _iterator():
        while capture.grab():
            yield capture.retrieve()[1]
    return tqdm(
        _iterator(),
        total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )

def main():
    '''Main function'''
    seed(42)
    predictor, _ = create_objects()
    cap = cv2.VideoCapture(opt.input_video)
    makedirs(opt.output, exist_ok=True)
    print(opt.output)
    for idx, frame in enumerate(frame_iter(cap)):
        frame = prepare_image(frame)
        prediction = predictor.predict(frame)
        cv2.imwrite(f'{opt.output}/{idx:08}.jpg', prediction)
    cap.release()

if __name__ == "__main__":
    main()
