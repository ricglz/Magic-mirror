from argparse import ArgumentParser

from afy.shared_arguments import add_predictor_arguments

def get_local_opt():
    parser = ArgumentParser()

    add_predictor_arguments(parser)

    parser.add_argument(
        '--input-video',
        type=str,
        required=True,
        help='Video to transform'
    )
    parser.add_argument(
        '--avatars',
        default='./avatars',
        help='path to avatars directory'
    )
    parser.add_argument(
        '--output',
        default='video.mp4',
        help='filename that will be given to the output video'
    )

    return parser.parse_args()

local_opt = get_local_opt()
