import argparse
import os

from copy import deepcopy
import torch

from evaluator.voc_evaluator import VOCAPIEvaluator
from evaluator.coco_evaluator import COCOAPIEvaluator
from evaluator.ourdataset_evaluator import OurDatasetEvaluator

# load transform
from dataset.ourdataset import our_class_labels
from dataset.data_augment import build_transform

# load some utils
from utils.misc import load_weight
from utils.misc import compute_flops

from models.detectors import build_model
from config import build_model_config, build_trans_config


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-Tutorial')
    # basic
    parser.add_argument('-size', '--img_size', default=640, type=int,
                        help='the max size of input image')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use cuda')

    # model
    parser.add_argument('-m', '--model', default='yolov1', type=str,
                        help='build yolo')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('-ct', '--conf_thresh', default=0.001, type=float,
                        help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.6, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=1000, type=int,
                        help='topk candidates for testing')
    parser.add_argument("--no_decode", action="store_true", default=False,
                        help="not decode in inference or yes")
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False,
                        help='fuse Conv & BN')

    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc.')
    # TTA
    parser.add_argument('-tta', '--test_aug', action='store_true', default=False,
                        help='use test augmentation.')

    return parser.parse_args()



def voc_test(model, data_dir, device, transform):
    evaluator = VOCAPIEvaluator(data_dir=data_dir,
                                device=device,
                                transform=transform,
                                display=True)

    # VOC evaluation
    evaluator.evaluate(model)


def coco_test(model, data_dir, device, transform, test=False):
    if test:
        # test-dev
        print('test on test-dev 2017')
        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        device=device,
                        testset=True,
                        transform=transform)

    else:
        # eval
        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        device=device,
                        testset=False,
                        transform=transform)

    # COCO evaluation
    evaluator.evaluate(model)


def our_test(model, data_dir, device, transform):
    evaluator = OurDatasetEvaluator(
        data_dir=data_dir,
        device=device,
        image_set='val',
        transform=transform)

    # WiderFace evaluation
    evaluator.evaluate(model)


if __name__ == '__main__':
    args = parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # dataset
    if args.dataset == 'voc':
        print('eval on voc ...')
        num_classes = 20
        data_dir = os.path.join(args.root, 'VOCdevkit')
    elif args.dataset == 'coco-val':
        print('eval on coco-val ...')
        num_classes = 80
        data_dir = os.path.join(args.root, 'COCO')
    elif args.dataset == 'coco-test':
        print('eval on coco-test-dev ...')
        num_classes = 80
        data_dir = os.path.join(args.root, 'COCO')
    elif args.dataset == 'ourdataset':
        print('eval on crowdhuman ...')
        num_classes = len(our_class_labels)
        data_dir = os.path.join(args.root, 'OurDataset')
    else:
        print('unknow dataset !! we only support voc, coco-val, coco-test !!!')
        exit(0)

    # config
    model_cfg = build_model_config(args)
    trans_cfg = build_trans_config(model_cfg['trans_type'])

    # build model
    model = build_model(args, model_cfg, device, num_classes, False)

    # load trained weight
    model = load_weight(model, args.weight, args.fuse_conv_bn)
    model.to(device).eval()

    # compute FLOPs and Params
    model_copy = deepcopy(model)
    model_copy.trainable = False
    model_copy.eval()
    compute_flops(
        model=model_copy,
        img_size=args.img_size, 
        device=device)
    del model_copy

    # transform
    transform = build_transform(args.img_size, trans_cfg, is_train=False)

    # evaluation
    with torch.no_grad():
        if args.dataset == 'voc':
            voc_test(model, data_dir, device, transform)
        elif args.dataset == 'coco-val':
            coco_test(model, data_dir, device, transform, test=False)
        elif args.dataset == 'coco-test':
            coco_test(model, data_dir, device, transform, test=True)
        elif args.dataset == 'ourdataset':
            our_test(model, data_dir, device, transform)
