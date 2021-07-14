'''Module to test using a video the local predictor'''
import cv2

from afy.local_arguments import local_opt
from afy.utils import Tee
from afy.predictor_local import PredictorLocal
from afy.helper_functions import load_images, prepare_image

log = Tee('./var/log/test_locally.log')

def create_objects():
    log('Creating Predictor')
    predictor = PredictorLocal(
        local_opt.swap_face,
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
    predictor, _ = create_objects()
    cap = cv2.VideoCapture(local_opt.input_video)
    fps = 24
    out = cv2.VideoWriter(
        'output.mp4',
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
