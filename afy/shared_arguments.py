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
    parser.add_argument(
        "--swap-face",
        action="store_true",
        help="Whether to perform face swap algorithm"
    )
    parser.add_argument(
        "--swapper",
        type=str,
        choices=['eds', 'triangulation', 'poisson'],
        default='poisson',
        help='Which swapper algorithm to use'
    )
    parser.add_argument(
        "--fsgan",
        action="store_true",
        help="Whether to use fsgan or not"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[256, 512],
        default=256,
        help="Resolution of the output img"
    )
    return parser
