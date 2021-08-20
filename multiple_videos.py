from argparse import ArgumentParser

import cv2
from tqdm import tqdm

window_names = ['Original', 'Close mouth', 'Open mouth']

VideoCapture = cv2.VideoCapture

def get_arguments():
    parser = ArgumentParser()
    for idx, name in enumerate(window_names):
        parser.add_argument(f'--video-{idx + 1}', type=str, help=name)

    return parser.parse_args()

def configure_windows():
    x_dimensions = [100, 400, 700]
    for window_name, x_dimension in zip(window_names, x_dimensions):
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
        cv2.moveWindow(window_name, x_dimension, 250)

def show_videos(cap_1: VideoCapture, cap_2: VideoCapture, cap_3: VideoCapture):
    while cap_1.isOpened():
        ret, frame_1 = cap_1.read()
        _, frame_2 = cap_2.read()
        _, frame_3 = cap_3.read()

        if not ret:
            break

        frames = [frame_1, frame_2, frame_3]
        for window_name, frame in zip(window_names, frames):
            cv2.imshow(window_name, cv2.resize(frame, (256, 256)))

        if cv2.waitKey(41) == 27:
            break

def main():
    arguments = get_arguments()
    configure_windows()

    cap_1 = VideoCapture(arguments.video_1)
    cap_2 = VideoCapture(arguments.video_2)
    cap_3 = VideoCapture(arguments.video_3)

    show_videos(cap_1, cap_2, cap_3)

    cap_1.release()
    cap_2.release()
    cap_3.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
