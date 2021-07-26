'''Main module to run camera prediction'''
from sys import platform as _platform
import os
import sys

import cv2
import numpy as np
import requests
import yaml

from afy.arguments import opt
from afy.helper_functions import load_images
from afy.utils import info, Tee, pad_img, resize, TicToc
from afy.videocaptureasync import VideoCaptureAsync
import afy.camera_selector as cam_selector

log = Tee('./var/log/cam_fomm.log')

# Where to split an array from face_alignment to separate each landmark
LANDMARK_SLICE_ARRAY = np.array([17, 22, 27, 31, 36, 42, 48, 60])

# if _platform == 'darwin':
#     if not opt.is_client:
#         info('\nOnly remote GPU mode is supported for Mac (use --is-client and --connect options to connect to the server)')
#         info('Standalone version will be available lately!\n')
#         sys.exit(1)

def is_new_frame_better(driving, predictor):
    global avatar_kp, display_string

    if avatar_kp is None:
        display_string = "No face detected in avatar."
        return False

    if predictor.get_start_frame() is None:
        display_string = "No frame to compare to."
        return True

    new_kp = predictor.get_frame_kp(driving)

    if new_kp is None:
        display_string = "No face found!"
        return False
    new_norm = (np.abs(avatar_kp - new_kp) ** 2).sum()
    old_norm = (np.abs(avatar_kp - predictor.get_start_frame_kp()) ** 2).sum()

    out_string = "{0} : {1}".format(int(new_norm * 100), int(old_norm * 100))
    display_string = out_string
    log(out_string)

def load_stylegan_avatar():
    url = "https://thispersondoesnotexist.com/image"
    r = requests.get(url, headers={'User-Agent': "My User Agent 1.0"}).content

    image = np.frombuffer(r, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = resize(image, (IMG_SIZE, IMG_SIZE))

    return image

def change_avatar(predictor, new_avatar):
    global avatar, avatar_kp
    avatar_kp = predictor.get_frame_kp(new_avatar)
    avatar = new_avatar
    predictor.set_source_image(avatar)

def kp_to_pixels(arr):
    '''Convert normalized landmark locations to screen pixels'''
    return ((arr + 1) * 127).astype(np.int32)

def draw_face_landmarks(img, face_kp, color=(20, 80, 255)):
    if face_kp is not None:
        img = cv2.polylines(img, np.split(kp_to_pixels(face_kp), LANDMARK_SLICE_ARRAY), False, color)

def print_help():
    info('\n\n=== Control keys ===')
    info('1-9: Change avatar')
    for i, fname in enumerate(avatar_names):
        key = i + 1
        name = fname.split('/')[-1]
        info(f'{key}: {name}')
    info('W: Zoom camera in')
    info('S: Zoom camera out')
    info('A: Previous avatar in folder')
    info('D: Next avatar in folder')
    info('Q: Get random avatar')
    info('X: Calibrate face pose')
    info('I: Show FPS')
    info('ESC: Quit')
    info('\nFull key list: https://github.com/alievk/avatarify#controls')
    info('\n\n')


def draw_fps(frame, fps, timing, x0=10, y0=20, ystep=30, fontsz=0.5, color=(255, 0, 0)):
    frame = frame.copy()
    cv2.putText(frame, f"FPS: {fps:.1f}", (x0, y0 + ystep * 0), 0, fontsz * IMG_SIZE / 256, color, 1)
    return frame

def draw_landmark_text(frame, thk=2, fontsz=0.5, color=(0, 0, 255)):
    frame = frame.copy()
    cv2.putText(frame, "ALIGN FACES", (60, 20), 0, fontsz * IMG_SIZE / 255, color, thk)
    cv2.putText(frame, "THEN PRESS X", (60, 245), 0, fontsz * IMG_SIZE / 255, color, thk)
    return frame


def draw_calib_text(frame, thk=2, fontsz=0.5, color=(0, 0, 255)):
    frame = frame.copy()
    cv2.putText(frame, "FIT FACE IN RECTANGLE", (40, 20), 0, fontsz * IMG_SIZE / 255, color, thk)
    cv2.putText(frame, "W - ZOOM IN", (60, 40), 0, fontsz * IMG_SIZE / 255, color, thk)
    cv2.putText(frame, "S - ZOOM OUT", (60, 60), 0, fontsz * IMG_SIZE / 255, color, thk)
    cv2.putText(frame, "THEN PRESS X", (60, 245), 0, fontsz * IMG_SIZE / 255, color, thk)
    return frame


def select_camera(config):
    cam_config = config['cam_config']
    cam_id = None

    if os.path.isfile(cam_config):
        with open(cam_config, 'r') as f:
            cam_config = yaml.load(f, Loader=yaml.FullLoader)
            cam_id = cam_config['cam_id']
    else:
        cam_frames = cam_selector.query_cameras(config['query_n_cams'])

        if cam_frames:
            if len(cam_frames) == 1:
                cam_id = list(cam_frames)[0]
            else:
                cam_id = cam_selector.select_camera(cam_frames, window="CLICK ON YOUR CAMERA")
            log(f"Selected camera {cam_id}")

            with open(cam_config, 'w') as f:
                yaml.dump({'cam_id': cam_id}, f)
        else:
            log("No cameras are available")

    return cam_id


if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    global display_string
    display_string = ""

    IMG_SIZE = 512

    log('Loading Predictor')
    predictor_args = {
        'checkpoint_path': opt.checkpoint,
        'config_path': opt.config,
        'swap_face': opt.swap_face,
        'swapper': opt.swapper,
        'verbose': opt.verbose,
    }
    if opt.is_worker:
        from afy import predictor_worker
        predictor_worker.run_worker(opt.in_port, opt.out_port)
        sys.exit(0)
    elif opt.is_client:
        from afy import predictor_remote
        try:
            predictor = predictor_remote.PredictorRemote(
                in_addr=opt.in_addr, out_addr=opt.out_addr,
                **predictor_args
            )
        except ConnectionError as err:
            log(err)
            sys.exit(1)
        predictor.start()
    else:
        from afy.utils import get_predictor
        predictor = get_predictor(predictor_args, opt.fsgan)

    cam_id = select_camera(config)

    if cam_id is None:
        sys.exit(1)

    cap = VideoCaptureAsync(cam_id)
    cap.start()

    avatars, avatar_names = load_images(opt)

    enable_vcam = not opt.no_stream

    ret, frame = cap.read()
    stream_img_size = frame.shape[1], frame.shape[0]

    if enable_vcam:
        if _platform in ['linux', 'linux2']:
            try:
                import pyfakewebcam
            except ImportError:
                log("pyfakewebcam is not installed.")
                sys.exit(1)

            stream = pyfakewebcam.FakeWebcam(f'/dev/video{opt.virt_cam}', *stream_img_size)
        else:
            enable_vcam = False

    cur_ava = 0
    avatar = None
    change_avatar(predictor, avatars[cur_ava])

    cv2.namedWindow('cam', cv2.WINDOW_GUI_NORMAL)
    cv2.moveWindow('cam', 100, 250)
    cv2.namedWindow('avatarify', cv2.WINDOW_GUI_NORMAL)
    cv2.moveWindow('avatarify', 600, 250)

    frame_proportion = 0.9
    frame_offset_x = 0
    frame_offset_y = 0

    fps_hist = []
    output_fps = []
    fps = 0
    show_fps = True

    print_help()

    try:
        while True:
            tt = TicToc()

            timing = {
                'preproc': 0,
                'predict': 0,
                'postproc': 0
            }

            tt.tic()

            ret, frame = cap.read()
            if not ret:
                log("Can't receive frame (stream end?). Exiting ...")
                break

            frame = resize(frame, (IMG_SIZE, IMG_SIZE))

            timing['preproc'] = tt.toc()

            tt.tic()
            out = predictor.predict(frame)
            timing['predict'] = tt.toc()

            tt.tic()

            key = cv2.waitKey(1)

            if cv2.getWindowProperty('cam', cv2.WND_PROP_VISIBLE) < 1.0:
                break
            elif cv2.getWindowProperty('avatarify', cv2.WND_PROP_VISIBLE) < 1.0:
                break

            if key == 27: # ESC
                break
            elif key == ord('d'):
                cur_ava += 1
                if cur_ava >= len(avatars):
                    cur_ava = 0
                change_avatar(predictor, avatars[cur_ava])
            elif key == ord('a'):
                cur_ava -= 1
                if cur_ava < 0:
                    cur_ava = len(avatars) - 1
                change_avatar(predictor, avatars[cur_ava])
            elif key == ord('w'):
                frame_proportion -= 0.05
                frame_proportion = max(frame_proportion, 0.1)
            elif key == ord('s'):
                frame_proportion += 0.05
                frame_proportion = min(frame_proportion, 1.0)
            elif key == ord('H'):
                frame_offset_x -= 1
            elif key == ord('h'):
                frame_offset_x -= 5
            elif key == ord('K'):
                frame_offset_x += 1
            elif key == ord('k'):
                frame_offset_x += 5
            elif key == ord('J'):
                frame_offset_y -= 1
            elif key == ord('j'):
                frame_offset_y -= 5
            elif key == ord('U'):
                frame_offset_y += 1
            elif key == ord('u'):
                frame_offset_y += 5
            elif key == ord('Z'):
                frame_offset_x = 0
                frame_offset_y = 0
                frame_proportion = 0.9
            elif key == ord('x'):
                predictor.reset_frames()
            elif key == ord('q'):
                try:
                    log('Loading StyleGAN avatar...')
                    avatar = load_stylegan_avatar()
                    change_avatar(predictor, avatar)
                except:
                    log('Failed to load StyleGAN avatar')
            elif key == ord('l'):
                try:
                    log('Reloading avatars...')
                    avatars, avatar_names = load_images(opt)
                    log("Images reloaded")
                except:
                    log('Image reload failed')
            elif key == ord('i'):
                show_fps = not show_fps
            elif 48 < key < 58:
                cur_ava = min(key - 49, len(avatars) - 1)
                change_avatar(predictor, avatars[cur_ava])
            elif key != -1:
                log(key)

            preview_frame = frame.copy()

            timing['postproc'] = tt.toc()

            if show_fps:
                preview_frame = draw_fps(preview_frame, fps, timing)

            cv2.imshow('cam', preview_frame)

            if out is not None:
                if not opt.no_pad:
                    out = pad_img(out, stream_img_size)

                if enable_vcam:
                    out = resize(out, stream_img_size)
                    stream.schedule_frame(out)

                cv2.imshow('avatarify', out)

            fps_hist.append(tt.toc(total=True))
            if len(fps_hist) == 10:
                fps = 10 / (sum(fps_hist) / 1000)
                output_fps.append(fps)
                fps_hist = []
    except KeyboardInterrupt:
        log("main: user interrupt")

    log("stopping camera")
    cap.stop()

    cv2.destroyAllWindows()

    if opt.is_client:
        log("stopping remote predictor")
        predictor.stop()

    log("main: exit")
    log(output_fps, min(output_fps), max(output_fps))
