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
from config import build_model_config, build_trans_config, build_dataset_config


def parse_args():
    parser = argparse.ArgumentParser(description='Real-time Object Detection LAB')
    # Basic setting
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
    parser.add_argument('--show', action='store_true', default=False,
                        help='show visualization')
    parser.add_argument('--gif', action='store_true', default=False, 
                        help='generate gif.')

    # Model setting
    parser.add_argument('-m', '--model', default='yolov1', type=str,
                        help='build yolo')
    parser.add_argument('-nc', '--num_classes', default=80, type=int,
                        help='number of classes.')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('-ct', '--conf_thresh', default=0.35, type=float,
                        help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.5, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=100, type=int,
                        help='topk candidates dets of each level before NMS')
    parser.add_argument("--deploy", action="store_true", default=False,
                        help="deploy mode or not")
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False,
                        help='fuse Conv & BN')
    parser.add_argument('--no_multi_labels', action='store_true', default=False,
                        help='Perform post-process with multi-labels trick.')
    parser.add_argument('--nms_class_agnostic', action='store_true', default=False,
                        help='Perform NMS operations regardless of category.')

    # Data setting
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc, crowdhuman, widerface.')

    return parser.parse_args()
                    

def detect(args,
           model, 
           device, 
           transform, 
           num_classes,
           class_names,
           class_indexs,
           mode='image'):
    # class color
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]
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
                x, _, ratio = transform(frame)
                x = x.unsqueeze(0).to(device)
                
                # inference
                t0 = time.time()
                outputs = model(x)
                scores = outputs['scores']
                labels = outputs['labels']
                bboxes = outputs['bboxes']
                t1 = time.time()
                print("Infer time: {:.1f} ms. ".format((t1 - t0) * 1000))

                # rescale bboxes
                bboxes = rescale_bboxes(bboxes, [orig_w, orig_h], ratio)

                # vis detection
                frame_vis = visualize(image=frame, 
                                      bboxes=bboxes,
                                      scores=scores, 
                                      labels=labels,
                                      class_colors=class_colors,
                                      class_names=class_names,
                                      class_indexs=class_indexs)
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
                x, _, ratio = transform(frame)
                x = x.unsqueeze(0).to(device)

                # inference
                t0 = time.time()
                outputs = model(x)
                scores = outputs['scores']
                labels = outputs['labels']
                bboxes = outputs['bboxes']
                t1 = time.time()
                print("Infer time: {:.1f} ms. ".format((t1 - t0) * 1000))

                # rescale bboxes
                bboxes = rescale_bboxes(bboxes, [orig_w, orig_h], ratio)

                # vis detection
                frame_vis = visualize(image=frame, 
                                      bboxes=bboxes,
                                      scores=scores, 
                                      labels=labels,
                                      class_colors=class_colors,
                                      class_names=class_names,
                                      class_indexs=class_indexs)

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
            x, _, ratio = transform(image)
            x = x.unsqueeze(0).to(device)

            # inference
            t0 = time.time()
            outputs = model(x)
            scores = outputs['scores']
            labels = outputs['labels']
            bboxes = outputs['bboxes']
            t1 = time.time()
            print("Infer time: {:.1f} ms. ".format((t1 - t0) * 1000))

            # rescale bboxes
            bboxes = rescale_bboxes(bboxes, [orig_w, orig_h], ratio)

            # vis detection
            img_processed = visualize(image=image, 
                                      bboxes=bboxes,
                                      scores=scores, 
                                      labels=labels,
                                      class_colors=class_colors,
                                      class_names=class_names,
                                      class_indexs=class_indexs)
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
    data_cfg  = build_dataset_config(args)
    
    ## Data info
    num_classes = data_cfg['num_classes']
    class_names = data_cfg['class_names']
    class_indexs = data_cfg['class_indexs']

    # build model
    model = build_model(args, model_cfg, device, args.num_classes, False)

    # load trained weight
    model = load_weight(model, args.weight, args.fuse_conv_bn)
    model.to(device).eval()

    # transform
    val_transform, trans_cfg = build_transform(args, trans_cfg, model_cfg['max_stride'], is_train=False)

    print("================= DETECT =================")
    # run
    detect(args         = args,
           mode         = args.mode,
           model        = model, 
           device       = device,
           transform    = val_transform,
           num_classes  = num_classes,
           class_names  = class_names,
           class_indexs = class_indexs,
           )


if __name__ == '__main__':
    run()
