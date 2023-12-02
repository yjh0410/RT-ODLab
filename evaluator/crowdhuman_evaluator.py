import os
import json
import time
import numpy as np

import torch


from dataset.crowdhuman import CrowdHumanDataset
from .crowdhuman_tools import compute_JI, compute_APMR


class CrowdHumanEvaluator():
    def __init__(self, data_dir, device, image_set='val', transform=None):
        """
        Args:
            data_dir (str): dataset root directory
            device: (int): CUDA or CPU.
            image_set: train or val.
            transform: used to preprocess inputs.
        """
        # ----------------- Basic parameters -----------------
        self.eval_source = os.path.join(data_dir, 'annotation_val.odgt')
        self.image_set = image_set
        self.transform = transform
        self.device = device
        self.evalDir = os.path.join('det_results', 'eval', 'CrowdHuman', time.strftime("%Y%S"))
        os.makedirs(self.evalDir, exist_ok=True)
        # ----------------- Metrics -----------------
        self.map = 0.
        self.mr = 0.
        self.ji = 0.
        # ----------------- Dataset -----------------
        self.dataset = CrowdHumanDataset(data_dir=data_dir, image_set=image_set)


    def boxes_dump(self, boxes):
        if boxes.shape[-1] == 7:
            result = [{'box':[round(i, 1) for i in box[:4]],
                    'score':round(float(box[4]), 5),
                    'tag':int(box[5]),
                    'proposal_num':int(box[6])} for box in boxes]
        elif boxes.shape[-1] == 6:
            result = [{'box':[round(i, 1) for i in box[:4].tolist()],
                    'score':round(float(box[4]), 5),
                    'tag':int(box[5])} for box in boxes]
        elif boxes.shape[-1] == 5:
            result = [{'box':[round(i, 1) for i in box[:4]],
                    'tag':int(box[4])} for box in boxes]
        else:
            raise ValueError('Unknown box dim.')
        return result


    @torch.no_grad()
    def inference(self, model):
        model.eval()
        all_result_dicts = []
        num_images = len(self.dataset)
        print('total number of images: %d' % (num_images))

        # start testing
        for index in range(num_images): # all the data in val2017
            if index % 500 == 0:
                print('[Eval: %d / %d]'%(index, num_images))

            # load an image
            img, img_id = self.dataset.pull_image(index)
            orig_h, orig_w, _ = img.shape

            # load a gt
            gt_bboxes, gt_labels = self.dataset.pull_anno(index)
            gt_bboxes = np.array(gt_bboxes)[..., :4]  # [N, 4]
            gt_tag = np.ones([gt_bboxes.shape[0], 1], dtype=gt_bboxes.dtype)
            gt_bboxes = np.concatenate([gt_bboxes, gt_tag], axis=-1)

            # preprocess
            x, _, deltas = self.transform(img)
            x = x.unsqueeze(0).to(self.device) / 255.
            
            # inference
            outputs = model(x)
            bboxes, scores, labels = outputs
            
            # rescale
            img_h, img_w = x.shape[-2:]
            bboxes[..., [0, 2]] = bboxes[..., [0, 2]] / (img_w - deltas[0]) * orig_w
            bboxes[..., [1, 3]] = bboxes[..., [1, 3]] / (img_h - deltas[1]) * orig_h
            
            # clip bboxes
            bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], a_min=0., a_max=orig_w)
            bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], a_min=0., a_max=orig_h)

            pd_tag = np.ones_like(scores)
            pd_bboxes = np.concatenate(
                [bboxes, scores[..., None], pd_tag[..., None]], axis=-1)

            # [x1, y1, x2, y2] -> [x1, y1, bw, bh]
            pd_bboxes[:, 2:4] -= pd_bboxes[:, :2]
            gt_bboxes[:, 2:4] -= gt_bboxes[:, :2]

            result_dict = dict(
                ID=img_id,
                height=int(orig_h),
                width=int(orig_w),
                dtboxes=self.boxes_dump(pd_bboxes.astype(np.float64)),
                gtboxes=self.boxes_dump(gt_bboxes.astype(np.float64))
                )
            all_result_dicts.append(result_dict)


        return all_result_dicts


    @torch.no_grad()
    def evaluate(self, model):
        # inference
        all_results = self.inference(model)

        # save json lines
        fpath = os.path.join(self.evalDir, 'dump-{}.json'.format('yolo_free'))
        with open(fpath,'w') as fid:
            for db in all_results:
                line = json.dumps(db)+'\n'
                fid.write(line)

        # evaluation
        eval_path = os.path.join(self.evalDir, 'eval-{}.json'.format('yolo_free'))
        eval_fid = open(eval_path,'w')
        res_line, JI = compute_JI.evaluation_all(fpath, 'box')
        for line in res_line:
            eval_fid.write(line+'\n')
        AP, MR = compute_APMR.compute_APMR(fpath, self.eval_source, 'box')
        line = 'AP:{:.4f}, MR:{:.4f}, JI:{:.4f}.'.format(AP, MR, JI)
        print(line)
        eval_fid.write(line+'\n')
        eval_fid.close()

        self.map = AP
        self.mr = MR
        self.ji = JI
