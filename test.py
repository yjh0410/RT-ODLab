import argparse
import cv2
import os
import time
import numpy as np
from copy import deepcopy
import torch

# load transform
from dataset.data_augment import build_transform

# load some utils
from utils.misc import build_dataset, load_weight
from utils.com_flops_params import FLOPs_and_Params
from utils.box_ops import rescale_bboxes
from utils import fuse_conv_bn

from models import build_model
from config import build_model_config, build_trans_config


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-Tutorial')

    # basic
    parser.add_argument('-size', '--img_size', default=416, type=int,
                        help='the max size of input image')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show the visulization results.')
    parser.add_argument('--save', action='store_true', default=False,
                        help='save the visulization results.')
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='use cuda.')
    parser.add_argument('--save_folder', default='det_results/', type=str,
                        help='Dir to save results')
    parser.add_argument('-vs', '--visual_threshold', default=0.3, type=float,
                        help='Final confidence threshold')
    parser.add_argument('-ws', '--window_scale', default=1.0, type=float,
                        help='resize window of cv2 for visualization.')

    # model
    parser.add_argument('-m', '--model', default='yolov1', type=str,
                        choices=['yolov1', 'yolov2', 'yolov3', 'yolov4'], help='build yolo')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('-ct', '--conf_thresh', default=0.1, type=float,
                        help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.5, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=100, type=int,
                        help='topk candidates for testing')
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False,
                        help='fuse conv and bn')
    parser.add_argument("--no_decode", action="store_true", default=False,
                        help="not decode in inference or yes")

    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc.')
    parser.add_argument('--min_box_size', default=8.0, type=float,
                        help='min size of target bounding box.')
    parser.add_argument('--mosaic', default=None, type=float,
                        help='mosaic augmentation.')
    parser.add_argument('--mixup', default=None, type=float,
                        help='mixup augmentation.')

    return parser.parse_args()


def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    
    if label is not None:
        # plot title bbox
        cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img


def visualize(img, 
              bboxes, 
              scores, 
              labels, 
              vis_thresh, 
              class_colors, 
              class_names, 
              class_indexs=None, 
              dataset_name='voc'):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            cls_id = int(labels[i])
            if dataset_name == 'coco':
                cls_color = class_colors[cls_id]
                cls_id = class_indexs[cls_id]
            else:
                cls_color = class_colors[cls_id]
                
            mess = '%s: %.2f' % (class_names[cls_id], scores[i])
            img = plot_bbox_labels(img, bbox, mess, cls_color, text_scale=ts)

    return img
        

@torch.no_grad()
def test(args,
         model, 
         device, 
         dataset,
         transforms=None,
         class_colors=None, 
         class_names=None, 
         class_indexs=None):
    num_images = len(dataset)
    save_path = os.path.join('det_results/', args.dataset, args.model)
    os.makedirs(save_path, exist_ok=True)

    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        image, _ = dataset.pull_image(index)

        orig_h, orig_w, _ = image.shape

        # prepare
        x, _, deltas = transforms(image)
        x = x.unsqueeze(0).to(device) / 255.

        t0 = time.time()
        # inference
        bboxes, scores, labels = model(x)
        print("detection time used ", time.time() - t0, "s")
        print(bboxes)
        
        # rescale bboxes
        origin_img_size = [orig_h, orig_w]
        cur_img_size = [*x.shape[-2:]]
        bboxes = rescale_bboxes(bboxes, origin_img_size, cur_img_size, deltas)

        # vis detection
        img_processed = visualize(
                            img=image,
                            bboxes=bboxes,
                            scores=scores,
                            labels=labels,
                            vis_thresh=args.visual_threshold,
                            class_colors=class_colors,
                            class_names=class_names,
                            class_indexs=class_indexs,
                            dataset_name=args.dataset)
        if args.show:
            h, w = img_processed.shape[:2]
            sw, sh = int(w*args.window_scale), int(h*args.window_scale)
            cv2.namedWindow('detection', 0)
            cv2.resizeWindow('detection', sw, sh)
            cv2.imshow('detection', img_processed)
            cv2.waitKey(0)

        if args.save:
            # save result
            cv2.imwrite(os.path.join(save_path, str(index).zfill(6) +'.jpg'), img_processed)


if __name__ == '__main__':
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

    # dataset and evaluator
    dataset, dataset_info, evaluator = build_dataset(args, trans_cfg, device, is_train=False)
    num_classes, class_names, class_indexs = dataset_info

    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]

    # build model
    model = build_model(args, model_cfg, device, num_classes, False)

    # load trained weight
    model = load_weight(model=model, path_to_ckpt=args.weight)
    model.to(device).eval()

    # compute FLOPs and Params
    model_copy = deepcopy(model)
    model_copy.trainable = False
    model_copy.eval()
    FLOPs_and_Params(
        model=model_copy,
        img_size=args.img_size, 
        device=device)
    del model_copy

    # fuse conv bn
    if args.fuse_conv_bn:
        print('fuse conv and bn ...')
        model = fuse_conv_bn.fuse_conv_bn(model)

    # transform
    transform = build_transform(args.img_size, trans_cfg, is_train=False)

    print("================= DETECT =================")
    # run
    test(args=args,
         model=model, 
         device=device, 
         dataset=dataset,
         transforms=transform,
         class_colors=class_colors,
         class_names=class_names,
         class_indexs=class_indexs
         )
