'''Module to test using a video the local predictor'''
from argparse import ArgumentParser

import cv2

from afy.local_arguments import local_opt
from afy.utils import Tee
from afy.predictor_local import PredictorLocal
from afy.helper_functions import load_images

log = Tee('./var/log/test_locally.log')

def create_objects():
    log('Creating Predictor')
    predictor = PredictorLocal(
        config_path=local_opt.config_path,
        checkpoint_path=local_opt.checkpoint_path,
    )
    log('Loading images')
    avatars, _ = load_images(local_opt)
    log('Setting source image')
    predictor.set_source_image(avatars[0])
    return predictor, avatars

def main():
    '''Main function'''
    predictor, avatars = create_objects()
    cap = cv2.VideoCapture(local_opt.input_video)
    ret, frame = cap.read()
    while ret:
        ret, frame = cap.read()

if __name__ == "__main__":
    main()
