import os
import cv2
import time
import imageio
import argparse
import numpy as np

import torch

from dataset.build import build_transform
from utils.vis_tools import plot_tracking
from utils.misc import load_weight
from utils.box_ops import rescale_bboxes

from config import build_model_config, build_trans_config

from models.detectors import build_model
from models.trackers import build_tracker

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def parse_args():
    parser = argparse.ArgumentParser(description='Tracking Task')

    # basic
    parser.add_argument('-size', '--img_size', default=640, type=int,
                        help='the max size of input image')
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='use cuda.')

    # data
    parser.add_argument('--mode', type=str, default='image',
                        help='image, video or camera')
    parser.add_argument('--path_to_img', type=str, default='dataset/demo/images/',
                        help='Dir to load images')
    parser.add_argument('--path_to_vid', type=str, default='dataset/demo/videos/',
                        help='Dir to load a video')
    parser.add_argument('--path_to_save', default='det_results/', type=str,
                        help='Dir to save results')
    parser.add_argument('--fps', type=int, default=30,
                        help='frame rate')
    parser.add_argument('--show', action='store_true', default=False, 
                        help='show results.')
    parser.add_argument('--save', action='store_true', default=False, 
                        help='save results.')
    parser.add_argument('--gif', action='store_true', default=False, 
                        help='generate gif.')

    # tracker
    parser.add_argument('-tk', '--tracker', default='byte_tracker', type=str,
                        help='build FreeTrack')
    parser.add_argument("--track_thresh", type=float, default=0.4, 
                        help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, 
                        help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, 
                        help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which \
                              aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10,
                        help='filter out tiny boxes')
    parser.add_argument("--mot20", default=False, action="store_true",
                        help="test mot20.")

    # detector
    parser.add_argument('-dt', '--model', default='yolov1', type=str,
                        help='build YOLO')
    parser.add_argument('-ns', '--num_classes', type=int, default=80,
                        help='number of object classes.')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('-ct', '--conf_thresh', default=0.3, type=float,
                        help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.5, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=100, type=int,
                        help='topk candidates for testing')
    parser.add_argument('-fcb', '--fuse_conv_bn', action='store_true', default=False,
                        help='fuse Conv & BN')

    return parser.parse_args()


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def run(args,
        tracker,
        detector,
        device, 
        transform):
    save_path = os.path.join(args.path_to_save, 'tracking', args.mode)
    os.makedirs(save_path, exist_ok=True)

    # ------------------------- Camera ----------------------------
    if args.mode == 'camera':
        print('use camera !!!')
        # Launch camera
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        frame_id = 0
        results = []

        # For saving
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_size = (640, 480)
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        save_video_name = os.path.join(save_path, cur_time+'.avi')
        fps = 15.0
        out = cv2.VideoWriter(save_video_name, fourcc, fps, save_size)
        print(save_video_name)
        image_list = []

        # start tracking
        while True:
            ret, frame = cap.read()
            if ret:
                if cv2.waitKey(1) == ord('q'):
                    break
                # ------------------------- Detection ---------------------------
                # preprocess
                x, _, deltas = transform(frame)
                x = x.unsqueeze(0).to(device) / 255.
                orig_h, orig_w, _ = frame.shape

                # detect
                t0 = time.time()
                bboxes, scores, labels = detector(x)
                print("=============== Frame-{} ================".format(frame_id))
                print("detect time: {:.1f} ms".format((time.time() - t0)*1000))

                # rescale bboxes
                origin_img_size = [orig_h, orig_w]
                cur_img_size = [*x.shape[-2:]]
                bboxes = rescale_bboxes(bboxes, origin_img_size, cur_img_size, deltas)

                # track
                t2 = time.time()
                if len(bboxes) > 0:
                    online_targets = tracker.update(scores, bboxes, labels)
                    online_xywhs = []
                    online_ids = []
                    online_scores = []
                    for t in online_targets:
                        xywh = t.xywh
                        tid = t.track_id
                        vertical = xywh[2] / xywh[3] > args.aspect_ratio_thresh
                        if xywh[2] * xywh[3] > args.min_box_area and not vertical:
                            online_xywhs.append(xywh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                            results.append(
                                f"{frame_id},{tid},{xywh[0]:.2f},{xywh[1]:.2f},{xywh[2]:.2f},{xywh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                                )
                    print("tracking time: {:.1f} ms".format((time.time() - t2)*1000))
                    
                    # plot tracking results
                    online_im = plot_tracking(
                        frame, online_xywhs, online_ids, frame_id=frame_id + 1, fps=1. / (time.time() - t0)
                    )
                else:
                    online_im = frame

                frame_resized = cv2.resize(online_im, save_size)
                out.write(frame_resized)

                if args.gif:
                    gif_resized = cv2.resize(online_im, (640, 480))
                    gif_resized_rgb = gif_resized[..., (2, 1, 0)]
                    image_list.append(gif_resized_rgb)

                # show results
                if args.show:
                    cv2.imshow('tracking', online_im)
                    ch = cv2.waitKey(1)
                    if ch == 27 or ch == ord("q") or ch == ord("Q"):
                        break
            else:
                break
            frame_id += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # generate GIF
        if args.gif:
            save_gif_path =  os.path.join(save_path, 'gif_files')
            os.makedirs(save_gif_path, exist_ok=True)
            save_gif_name = os.path.join(save_gif_path, '{}.gif'.format(cur_time))
            print('generating GIF ...')
            imageio.mimsave(save_gif_name, image_list, fps=fps)
            print('GIF done: {}'.format(save_gif_name))

    # ------------------------- Video ---------------------------
    elif args.mode == 'video':
        # read a video
        video = cv2.VideoCapture(args.path_to_vid)
        fps = video.get(cv2.CAP_PROP_FPS)
        
        # For saving
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_size = (640, 480)
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        save_video_name = os.path.join(save_path, cur_time+'.avi')
        out = cv2.VideoWriter(save_video_name, fourcc, fps, save_size)
        print(save_video_name)
        image_list = []

        # start tracking
        frame_id = 0
        results = []
        while(True):
            ret, frame = video.read()
            
            if ret:
                # ------------------------- Detection ---------------------------
                # preprocess
                x, _, deltas = transform(frame)
                x = x.unsqueeze(0).to(device) / 255.
                orig_h, orig_w, _ = frame.shape

                # detect
                t0 = time.time()
                bboxes, scores, labels = detector(x)
                print("=============== Frame-{} ================".format(frame_id))
                print("detect time: {:.1f} ms".format((time.time() - t0)*1000))

                # rescale bboxes
                origin_img_size = [orig_h, orig_w]
                cur_img_size = [*x.shape[-2:]]
                bboxes = rescale_bboxes(bboxes, origin_img_size, cur_img_size, deltas)

                # track
                t2 = time.time()
                if len(bboxes) > 0:
                    online_targets = tracker.update(scores, bboxes, labels)
                    online_xywhs = []
                    online_ids = []
                    online_scores = []
                    for t in online_targets:
                        xywh = t.xywh
                        tid = t.track_id
                        vertical = xywh[2] / xywh[3] > args.aspect_ratio_thresh
                        if xywh[2] * xywh[3] > args.min_box_area and not vertical:
                            online_xywhs.append(xywh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                            results.append(
                                f"{frame_id},{tid},{xywh[0]:.2f},{xywh[1]:.2f},{xywh[2]:.2f},{xywh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                                )
                    print("tracking time: {:.1f} ms".format((time.time() - t2)*1000))
                    
                    # plot tracking results
                    online_im = plot_tracking(
                        frame, online_xywhs, online_ids, frame_id=frame_id + 1, fps=1. / (time.time() - t0)
                    )
                else:
                    online_im = frame

                frame_resized = cv2.resize(online_im, save_size)
                out.write(frame_resized)

                if args.gif:
                    gif_resized = cv2.resize(online_im, (640, 480))
                    gif_resized_rgb = gif_resized[..., (2, 1, 0)]
                    image_list.append(gif_resized_rgb)

                # show results
                if args.show:
                    cv2.imshow('tracking', online_im)
                    ch = cv2.waitKey(1)
                    if ch == 27 or ch == ord("q") or ch == ord("Q"):
                        break
            else:
                break
            frame_id += 1

        video.release()
        out.release()
        cv2.destroyAllWindows()

        # generate GIF
        if args.gif:
            save_gif_path =  os.path.join(save_path, 'gif_files')
            os.makedirs(save_gif_path, exist_ok=True)
            save_gif_name = os.path.join(save_gif_path, '{}.gif'.format(cur_time))
            print('generating GIF ...')
            imageio.mimsave(save_gif_name, image_list, fps=fps)
            print('GIF done: {}'.format(save_gif_name))

    # ------------------------- Image ----------------------------
    elif args.mode == 'image':
        files = get_image_list(args.path_to_img)
        files.sort()

        # For saving
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_size = (640, 480)
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        save_video_name = os.path.join(save_path, cur_time+'.avi')
        out = cv2.VideoWriter(save_video_name, fourcc, fps, save_size)
        print(save_video_name)
        image_list = []

        # start tracking
        frame_id = 0
        results = []
        for frame_id, img_path in enumerate(files, 1):
            image = cv2.imread(os.path.join(img_path))
            # preprocess
            x, _, deltas = transform(image)
            x = x.unsqueeze(0).to(device) / 255.
            orig_h, orig_w, _ = image.shape

            # detect
            t0 = time.time()
            bboxes, scores, labels = detector(x)
            print("=============== Frame-{} ================".format(frame_id))
            print("detect time: {:.1f} ms".format((time.time() - t0)*1000))

            # rescale bboxes
            origin_img_size = [orig_h, orig_w]
            cur_img_size = [*x.shape[-2:]]
            bboxes = rescale_bboxes(bboxes, origin_img_size, cur_img_size, deltas)

            # track
            t2 = time.time()
            if len(bboxes) > 0:
                online_targets = tracker.update(scores, bboxes, labels)
                online_xywhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    xywh = t.xywh
                    tid = t.track_id
                    vertical = xywh[2] / xywh[3] > args.aspect_ratio_thresh
                    if xywh[2] * xywh[3] > args.min_box_area and not vertical:
                        online_xywhs.append(xywh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{xywh[0]:.2f},{xywh[1]:.2f},{xywh[2]:.2f},{xywh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                            )
                print("tracking time: {:.1f} ms".format((time.time() - t2)*1000))
                
                # plot tracking results
                online_im = plot_tracking(
                    image, online_xywhs, online_ids, frame_id=frame_id + 1, fps=1. / (time.time() - t0)
                )
            else:
                online_im = frame

            frame_resized = cv2.resize(online_im, save_size)
            out.write(frame_resized)

            if args.gif:
                gif_resized = cv2.resize(online_im, (640, 480))
                gif_resized_rgb = gif_resized[..., (2, 1, 0)]
                image_list.append(gif_resized_rgb)

            # show results
            if args.show:
                cv2.imshow('tracking', online_im)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break

            frame_id += 1

        cv2.destroyAllWindows()
        out.release()
        cv2.destroyAllWindows()

        # generate GIF
        if args.gif:
            save_gif_path =  os.path.join(save_path, 'gif_files')
            os.makedirs(save_gif_path, exist_ok=True)
            save_gif_name = os.path.join(save_gif_path, '{}.gif'.format(cur_time))
            print('generating GIF ...')
            imageio.mimsave(save_gif_name, image_list, fps=fps)
            print('GIF done: {}'.format(save_gif_name))


if __name__ == '__main__':
    args = parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    np.random.seed(0)

    # config
    model_cfg = build_model_config(args)
    trans_cfg = build_trans_config(model_cfg['trans_type'])

    # transform
    transform = build_transform(args.img_size, trans_cfg, is_train=False)

    # ---------------------- General Object Detector ----------------------
    detector = build_model(args, model_cfg, device, args.num_classes, False)

    ## load trained weight
    detector = load_weight(detector, args.weight, args.fuse_conv_bn)
    detector.to(device).eval()
    
    # ---------------------- General Object Tracker ----------------------
    tracker = build_tracker(args)

    # run
    run(args=args,
        tracker=tracker,
        detector=detector, 
        device=device,
        transform=transform)
