import cv2
import h5py
import os
import argparse
import numpy as np
import sys

sys.path.append('..')
from voc import VOCDetection
from coco import COCODataset

# ---------------------- Opt ----------------------
parser = argparse.ArgumentParser(description='Cache-Dataset')
parser.add_argument('-d', '--dataset', default='voc',
                    help='coco, voc, widerface, crowdhuman')
parser.add_argument('--root', default='/Users/liuhaoran/Desktop/python_work/object-detection/dataset/',
                    help='data root')
parser.add_argument('-size', '--img_size', default=640, type=int,
                    help='input image size.')
parser.add_argument('--mosaic', default=None, type=float,
                    help='mosaic augmentation.')
parser.add_argument('--mixup', default=None, type=float,
                    help='mixup augmentation.')
parser.add_argument('--keep_ratio', action="store_true", default=False,
                    help='keep aspect ratio.')
parser.add_argument('--show', action="store_true", default=False,
                    help='keep aspect ratio.')

args = parser.parse_args()


# ---------------------- Build Dataset ----------------------
if args.dataset == 'voc':
    root = os.path.join(args.root, 'VOCdevkit')
    dataset = VOCDetection(args.img_size, root)
elif args.dataset == 'coco':
    root = os.path.join(args.root, 'COCO')
    dataset = COCODataset(args.img_size, args.root)
print('Data length: ', len(dataset))


# ---------------------- Main Process ----------------------
cached_image = []
dataset_size = len(dataset)
for i in range(len(dataset)):
    if i % 5000 == 0:
        print("[{} / {}]".format(i, dataset_size))
    # load an image
    image, image_id = dataset.pull_image(i)
    orig_h, orig_w, _ = image.shape

    # resize image
    if args.keep_ratio:
        r = args.img_size / max(orig_h, orig_w)
        if r != 1: 
            interp = cv2.INTER_LINEAR
            new_size = (int(orig_w * r), int(orig_h * r))
            image = cv2.resize(image, new_size, interpolation=interp)
    else:
        image = cv2.resize(image, (int(args.img_size), int(args.img_size)))

    cached_image.append(image)
    if args.show:
        cv2.imshow('image', image)
        # cv2.imwrite(str(i)+'.jpg', img)
        cv2.waitKey(0)

save_path = "dataset/cache/"
os.makedirs(save_path, exist_ok=True)
np.save(save_path + '{}_train_images.npy'.format(args.dataset), cached_image)
