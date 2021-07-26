'''Module to test using a video the local predictor'''
from random import seed
import cv2

from afy.local_arguments import local_opt as opt
from afy.utils import Tee, get_predictor
from afy.helper_functions import load_images, prepare_image

log = Tee('./var/log/test_locally.log')

def create_objects():
    log('Creating Predictor')
    predictor = get_predictor(opt, opt.fsgan)
    log('Loading images')
    avatars, _ = load_images(opt)
    log('Setting source image')
    predictor.set_source_image(avatars[0])
    return predictor, avatars

def main():
    '''Main function'''
    seed(42)
    predictor, _ = create_objects()
    cap = cv2.VideoCapture(opt.input_video)
    fps = 24
    out = cv2.VideoWriter(
        opt.output,
        cv2.VideoWriter_fourcc(*'MP4V'),
        fps,
        predictor.output_size,
    )
    ret, frame = cap.read()
    while ret:
        frame = prepare_image(frame)
        prediction = predictor.predict(frame)
        out.write(prediction)
        ret, frame = cap.read()
    cap.release()

if __name__ == "__main__":
    main()
