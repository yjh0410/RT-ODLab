import os
import json


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='COCO-Dataset')

    # --------------- opt parameters ---------------
    parser.add_argument('--root', default='/Users/liuhaoran/Desktop/python_work/object-detection/dataset/COCO/',
                        help='data root')
    parser.add_argument('--image_set', type=str, default='val',
                        help='augmentation type')
    parser.add_argument('--task', type=str, default='det',
                        help='augmentation type')
    
    args = parser.parse_args()

    # --------------- load json ---------------
    if args.task == 'det':
        task_prefix = 'instances_{}2017.json'
        clean_task_prefix = 'instances_{}2017_clean.json'
    elif args.task == 'pos':
        task_prefix = 'person_keypoints_{}2017.json'
        clean_task_prefix = 'person_keypoints_{}2017_clean.json'
    else:
        raise NotImplementedError('Unkown task !')
    
    json_path = os.path.join(args.root, 'annotations', task_prefix.format(args.image_set))

    clean_json_file = dict()
    with open(json_path, 'r') as file:
        json_file = json.load(file)
        # json_file is a Dict: dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
        clean_json_file['info'] = json_file['info'] 
        clean_json_file['licenses'] = json_file['licenses']
        clean_json_file['categories'] = json_file['categories']

        images_list = json_file['images']
        annots_list = json_file['annotations']
        num_images = len(images_list)

        # -------------- Filter annotations --------------
        print("Processing annotations ...")
        valid_image_ids = []
        clean_annots_list = [] 
        for i, anno in enumerate(annots_list):
            if i % 5000 == 0:
                print("[{}] / [{}] ...".format(i, len(annots_list)))
            x1, y1, bw, bh = anno['bbox']
            if bw > 0 and bh > 0:
                clean_annots_list.append(anno)
                if anno['image_id'] not in valid_image_ids:
                    valid_image_ids.append(anno['image_id'])
        print("Valid number of images: ", len(valid_image_ids))
        print("Valid number of annots: ", len(clean_annots_list))
        print("Original number of annots: ", len(annots_list))

        # -------------- Filter images --------------
        print("Processing images ...")
        clean_images_list = []
        for i in range(num_images):
            if args.image_set == 'train' and i % 5000 == 0:
                print("[{}] / [{}] ...".format(i, num_images))
            if args.image_set == 'val' and i % 500 == 0:
                print("[{}] / [{}] ...".format(i, num_images))
            
            # A single image dict
            image_dict = images_list[i]
            image_id = image_dict['id']

            if image_id in valid_image_ids:
                clean_images_list.append(image_dict)

        print('Number of images after cleaning: ', len(clean_images_list))
        print('Number of annotations after cleaning: ', len(clean_annots_list))

        clean_json_file['images'] = clean_images_list
        clean_json_file['annotations'] = clean_annots_list
    
    # --------------- Save filterd json file ---------------
    new_json_path = os.path.join(args.root, 'annotations', clean_task_prefix.format(args.image_set))
    with open(new_json_path, 'w') as f:
        json.dump(clean_json_file, f)
