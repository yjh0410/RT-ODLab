import argparse
import json
import os
import os.path as osp
from PIL import Image


def parse_wider_gt(dets_file_name):
    # -----------------------------------------------------------------------------------------
    '''
      Parse the FDDB-format detection output file:
        - first line is image file name
        - second line is an integer, for `n` detections in that image
        - next `n` lines are detection coordinates
        - again, next line is image file name
        - detections are [x y width height score]
      Returns a dict: {'img_filename': detections as a list of arrays}
    '''
    fid = open(dets_file_name, 'r')

    # Parsing the FDDB-format detection output txt file
    img_flag = True
    numdet_flag = False
    start_det_count = False
    det_count = 0
    numdet = -1

    det_dict = {}
    img_file = ''

    for line in fid:
        line = line.strip()

        if line == '0 0 0 0 0 0 0 0 0 0':
            if det_count == numdet - 1:
                start_det_count = False
                det_count = 0
                img_flag = True  # next line is image file
                numdet_flag = False
                numdet = -1
                det_dict.pop(img_file)
            continue

        if img_flag:
            # Image filename
            img_flag = False
            numdet_flag = True
            # print('Img file: ' + line)
            img_file = line
            det_dict[img_file] = []  # init detections list for image
            continue

        if numdet_flag:
            # next line after image filename: number of detections
            numdet = int(line)
            numdet_flag = False
            if numdet > 0:
                start_det_count = True  # start counting detections
                det_count = 0
            else:
                # no detections in this image
                img_flag = True  # next line is another image file
                numdet = -1

            # print 'num det: ' + line
            continue

        if start_det_count:
            # after numdet, lines are detections
            detection = [float(x) for x in line.split()]  # split on whitespace
            det_dict[img_file].append(detection)
            # print 'Detection: %s' % line
            det_count += 1

        if det_count == numdet:
            start_det_count = False
            det_count = 0
            img_flag = True  # next line is image file
            numdet_flag = False
            numdet = -1

    return det_dict


def convert_wider_annots(args):
    """Convert from WIDER FDDB-style format to COCO bounding box"""

    subset = ['train', 'val'] if args.subset == 'all' else [args.subset]
    outdir = os.path.join(args.datadir, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    categories = [{"id": 1, "name": 'face'}]
    for sset in subset:
        print(f'Processing subset {sset}')
        out_json_name = osp.join(outdir, f'{sset}.json')
        data_dir = osp.join(args.datadir, f'WIDER_{sset}', 'images')
        img_id = 0
        ann_id = 0
        cat_id = 1

        ann_dict = {}
        images = []
        annotations = []
        ann_file = os.path.join(args.datadir, 'wider_face_split', f'wider_face_{sset}_bbx_gt.txt')
        wider_annot_dict = parse_wider_gt(ann_file)  # [im-file] = [[x,y,w,h], ...]

        for filename in wider_annot_dict.keys():
            if len(images) % 100 == 0:
                print("Processed %s images, %s annotations" % (
                    len(images), len(annotations)))

            image = {}
            image['id'] = img_id
            img_id += 1
            im = Image.open(os.path.join(data_dir, filename))
            image['width'] = im.height
            image['height'] = im.width
            image['file_name'] = filename
            images.append(image)

            for gt_bbox in wider_annot_dict[filename]:
                ann = {}
                ann['id'] = ann_id
                ann_id += 1
                ann['image_id'] = image['id']
                ann['segmentation'] = []
                ann['category_id'] = cat_id  # 1:"face" for WIDER
                ann['iscrowd'] = 0
                ann['area'] = gt_bbox[2] * gt_bbox[3]
                ann['boxes'] = gt_bbox
                ann['bbox'] = gt_bbox[:4]
                annotations.append(ann)

        ann_dict['images'] = images
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations
        print("Num categories: %s" % len(categories))
        print("Num images: %s" % len(images))
        print("Num annotations: %s" % len(annotations))
        with open(out_json_name, 'w', encoding='utf8') as outfile:
            json.dump(ann_dict, outfile, indent=4, sort_keys=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument(
        '-d', '--datadir', help="dir to widerface", default='data/widerface', type=str)

    parser.add_argument(
        '-s', '--subset', help="which subset to convert", default='all', choices=['all', 'train', 'val'], type=str)

    parser.add_argument(
        '-o', '--outdir', help="where to output the annotation file, default same as data dir", default='annotations')

    args = parser.parse_args()

    convert_wider_annots(args)