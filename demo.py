import argparse
import cv2
import os
import time
import numpy as np
import imageio

import torch

# load transform
from dataset.build import build_transform

# load some utils
from utils.misc import load_weight
from utils.box_ops import rescale_bboxes
from utils.vis_tools import visualize

from models.detectors import build_model
from config import build_model_config, build_trans_config



def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Demo')

    # basic
    parser.add_argument('-size', '--img_size', default=640, type=int,
                        help='the max size of input image')
    parser.add_argument('--mosaic', default=None, type=float,
                        help='mosaic augmentation.')
    parser.add_argument('--mixup', default=None, type=float,
                        help='mixup augmentation.')
    parser.add_argument('--mode', default='image',
                        type=str, help='Use the data from image, video or camera')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use cuda')
    parser.add_argument('--path_to_img', default='dataset/demo/images/',
                        type=str, help='The path to image files')
    parser.add_argument('--path_to_vid', default='dataset/demo/videos/',
                        type=str, help='The path to video files')
    parser.add_argument('--path_to_save', default='det_results/demos/',
                        type=str, help='The path to save the detection results')
    parser.add_argument('-vt', '--vis_thresh', default=0.4, type=float,
                        help='Final confidence threshold for visualization')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show visualization')
    parser.add_argument('--gif', action='store_true', default=False, 
                        help='generate gif.')

    # model
    parser.add_argument('-m', '--model', default='yolov1', type=str,
                        help='build yolo')
    parser.add_argument('-nc', '--num_classes', default=80, type=int,
                        help='number of classes.')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('-ct', '--conf_thresh', default=0.1, type=float,
                        help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.5, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=100, type=int,
                        help='topk candidates for testing')
    parser.add_argument("--deploy", action="store_true", default=False,
                        help="deploy mode or not")
    parser.add_argument('--fuse_repconv', action='store_true', default=False,
                        help='fuse RepConv')
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False,
                        help='fuse Conv & BN')
    parser.add_argument('--nms_class_agnostic', action='store_true', default=False,
                        help='Perform NMS operations regardless of category.')

    return parser.parse_args()
                    

def detect(args,
           model, 
           device, 
           transform, 
           vis_thresh, 
           mode='image'):
    # class color
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(80)]
    save_path = os.path.join(args.path_to_save, mode)
    os.makedirs(save_path, exist_ok=True)

    # ------------------------- Camera ----------------------------
    if mode == 'camera':
        print('use camera !!!')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_size = (640, 480)
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        save_video_name = os.path.join(save_path, cur_time+'.avi')
        fps = 15.0
        out = cv2.VideoWriter(save_video_name, fourcc, fps, save_size)
        print(save_video_name)
        image_list = []

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            ret, frame = cap.read()
            if ret:
                if cv2.waitKey(1) == ord('q'):
                    break
                orig_h, orig_w, _ = frame.shape

                # prepare
                x, _, deltas = transform(frame)
                x = x.unsqueeze(0).to(device) / 255.
                
                # inference
                t0 = time.time()
                bboxes, scores, labels = model(x)
                t1 = time.time()
                print("detection time used ", t1-t0, "s")

                # rescale bboxes
                origin_img_size = [orig_h, orig_w]
                cur_img_size = [*x.shape[-2:]]
                bboxes = rescale_bboxes(bboxes, origin_img_size, cur_img_size, deltas)

                # vis detection
                frame_vis = visualize(img=frame, 
                                      bboxes=bboxes,
                                      scores=scores, 
                                      labels=labels,
                                      class_colors=class_colors,
                                      vis_thresh=vis_thresh)
                frame_resized = cv2.resize(frame_vis, save_size)
                out.write(frame_resized)

                if args.gif:
                    gif_resized = cv2.resize(frame, (640, 480))
                    gif_resized_rgb = gif_resized[..., (2, 1, 0)]
                    image_list.append(gif_resized_rgb)

                if args.show:
                    cv2.imshow('detection', frame_resized)
                    cv2.waitKey(1)
            else:
                break
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
    elif mode == 'video':
        video = cv2.VideoCapture(args.path_to_vid)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_size = (640, 480)
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        save_video_name = os.path.join(save_path, cur_time+'.avi')
        fps = 15.0
        out = cv2.VideoWriter(save_video_name, fourcc, fps, save_size)
        print(save_video_name)
        image_list = []

        while(True):
            ret, frame = video.read()
            
            if ret:
                # ------------------------- Detection ---------------------------
                orig_h, orig_w, _ = frame.shape

                # prepare
                x, _, deltas = transform(frame)
                x = x.unsqueeze(0).to(device) / 255.

                # inference
                t0 = time.time()
                bboxes, scores, labels = model(x)
                t1 = time.time()
                print("detection time used ", t1-t0, "s")

                # rescale bboxes
                origin_img_size = [orig_h, orig_w]
                cur_img_size = [*x.shape[-2:]]
                bboxes = rescale_bboxes(bboxes, origin_img_size, cur_img_size, deltas)

                # vis detection
                frame_vis = visualize(img=frame, 
                                      bboxes=bboxes,
                                      scores=scores, 
                                      labels=labels,
                                      class_colors=class_colors,
                                      vis_thresh=vis_thresh)

                frame_resized = cv2.resize(frame_vis, save_size)
                out.write(frame_resized)

                if args.gif:
                    gif_resized = cv2.resize(frame, (640, 480))
                    gif_resized_rgb = gif_resized[..., (2, 1, 0)]
                    image_list.append(gif_resized_rgb)

                if args.show:
                    cv2.imshow('detection', frame_resized)
                    cv2.waitKey(1)
            else:
                break
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
    elif mode == 'image':
        for i, img_id in enumerate(os.listdir(args.path_to_img)):
            image = cv2.imread((args.path_to_img + '/' + img_id), cv2.IMREAD_COLOR)
            orig_h, orig_w, _ = image.shape

            # prepare
            x, _, deltas = transform(image)
            x = x.unsqueeze(0).to(device) / 255.

            # inference
            t0 = time.time()
            bboxes, scores, labels = model(x)
            t1 = time.time()
            print("detection time used ", t1-t0, "s")

            # rescale bboxes
            origin_img_size = [orig_h, orig_w]
            cur_img_size = [*x.shape[-2:]]
            bboxes = rescale_bboxes(bboxes, origin_img_size, cur_img_size, deltas)

            # vis detection
            img_processed = visualize(img=image, 
                                      bboxes=bboxes,
                                      scores=scores, 
                                      labels=labels,
                                      class_colors=class_colors,
                                      vis_thresh=vis_thresh)
            cv2.imwrite(os.path.join(save_path, str(i).zfill(6)+'.jpg'), img_processed)
            if args.show:
                cv2.imshow('detection', img_processed)
                cv2.waitKey(0)


def run():
    args = parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    model_cfg = build_model_config(args)
    trans_cfg = build_trans_config(model_cfg['trans_type'])

    # build model
    model = build_model(args, model_cfg, device, args.num_classes, False)

    # load trained weight
    model = load_weight(model, args.weight, args.fuse_conv_bn)
    model.to(device).eval()

    # transform
    val_transform, trans_cfg = build_transform(args, trans_cfg, model_cfg['max_stride'], is_train=False)

    print("================= DETECT =================")
    # run
    detect(args=args,
           model=model, 
            device=device,
            transform=val_transform,
            mode=args.mode,
            vis_thresh=args.vis_thresh)


if __name__ == '__main__':
    run()
