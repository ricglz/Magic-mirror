from argparse import ArgumentParser, Namespace
from os import path

import cv2
from tqdm import tqdm

window_names = ['Original', 'Close mouth', 'Open mouth']
algorithms = ('articulated', 'eds', 'fsgan', 'poisson', 'triangulation')

VideoCapture = cv2.VideoCapture

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--individual', type=str)
    parser.add_argument('--resolution', type=int, choices=(256, 512))
    parser.add_argument('--algorithm', type=str, choices=algorithms)

    return parser.parse_args()

def get_videos(args: Namespace):
    base_path = path.join('videos', args.individual)
    video_1 = path.join(base_path, 'original.mp4')
    algorithm_filename = f'{args.algorithm}.mp4'
    # resolution_path = path.join(base_path, f'{args.resolution}x{args.resolution}')
    video_2 = path.join(base_path, '0', algorithm_filename)
    video_3 = path.join(base_path, '1', algorithm_filename)
    return video_1, video_2, video_3

def configure_windows():
    x_dimensions = [100, 400, 700]
    for window_name, x_dimension in zip(window_names, x_dimensions):
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
        cv2.moveWindow(window_name, x_dimension, 250)

def show_videos(
    cap_1: VideoCapture,
    cap_2: VideoCapture,
    cap_3: VideoCapture,
    resolution: int,
):
    desired_size = resolution, resolution
    while cap_1.isOpened():
        ret_1, frame_1 = cap_1.read()
        ret_2, frame_2 = cap_2.read()
        ret_3, frame_3 = cap_3.read()

        if not all((ret_1, ret_2, ret_3)):
            break

        frames = [frame_1, frame_2, frame_3]
        for window_name, frame in zip(window_names, frames):
            cv2.imshow(window_name, cv2.resize(frame, desired_size))

        if cv2.waitKey(20) == 27:
            break

def main():
    configure_windows()
    arguments = get_arguments()
    video_1, video_2, video_3 = get_videos(arguments)

    cap_1 = VideoCapture(video_1)
    cap_2 = VideoCapture(video_2)
    cap_3 = VideoCapture(video_3)

    show_videos(cap_1, cap_2, cap_3, arguments.resolution)

    cap_1.release()
    cap_2.release()
    cap_3.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
