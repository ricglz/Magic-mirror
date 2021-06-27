from argparse import ArgumentParser

def add_predictor_arguments(parser: ArgumentParser):
    parser.add_argument("--config", help="path to config")
    parser.add_argument(
        "--checkpoint",
        default='vox-cpk.pth.tar',
        help="path to checkpoint to restore"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional information"
    )
    return parser

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
        "--avatars",
        default="./avatars",
        help="path to avatars directory"
    )

    return parser.parse_args()

local_opt = get_local_opt()
