import os
import sys
import json
import math
import argparse
from multiprocessing import Queue, Process

from tqdm import tqdm
import numpy as np

from .JIToolkits.JI_tools import compute_matching, get_ignores
sys.path.insert(0, '../')


# ---------------------------------- JI Evaluation functions ----------------------------------
def evaluation_all(path, target_key, nr_procs=10):
    records = load_json_lines(path)
    res_line = []
    res_JI = []
    for i in range(10):
        score_thr = 1e-1 * i
        total = len(records)
        stride = math.ceil(total / nr_procs)
        result_queue = Queue(10000)
        results, procs = [], []
        for i in range(nr_procs):
            start = i*stride
            end = np.min([start+stride,total])
            sample_data = records[start:end]
            p = Process(target= compute_JI_with_ignore, args=(result_queue, sample_data, score_thr, target_key))
            p.start()
            procs.append(p)
        tqdm.monitor_interval = 0
        pbar = tqdm(total=total, leave = False, ascii = True)
        for i in range(total):
            t = result_queue.get()
            results.append(t)
            pbar.update(1)
        for p in procs:
            p.join()
        pbar.close()
        line, mean_ratio = gather(results)
        line = 'score_thr:{:.1f}, {}'.format(score_thr, line)
        print(line)
        res_line.append(line)
        res_JI.append(mean_ratio)
    return res_line, max(res_JI)


def compute_JI_with_ignore(result_queue, records, score_thr, target_key, bm_thresh=0.5):
    for record in records:
        gt_boxes = load_bboxes(record, 'gtboxes', target_key, 'tag')
        gt_boxes[:,2:4] += gt_boxes[:,:2]
        gt_boxes = clip_boundary(gt_boxes, record['height'], record['width'])
        dt_boxes = load_bboxes(record, 'dtboxes', target_key, 'score')
        dt_boxes[:,2:4] += dt_boxes[:,:2]
        dt_boxes = clip_boundary(dt_boxes, record['height'], record['width'])
        keep = dt_boxes[:, -1] > score_thr
        dt_boxes = dt_boxes[keep][:, :-1]

        gt_tag = np.array(gt_boxes[:,-1]!=-1)
        matches = compute_matching(dt_boxes, gt_boxes[gt_tag, :4], bm_thresh)
        # get the unmatched_indices
        matched_indices = np.array([j for (j,_) in matches])
        unmatched_indices = list(set(np.arange(dt_boxes.shape[0])) - set(matched_indices))
        num_ignore_dt = get_ignores(dt_boxes[unmatched_indices], gt_boxes[~gt_tag, :4], bm_thresh)
        matched_indices = np.array([j for (_,j) in matches])
        unmatched_indices = list(set(np.arange(gt_boxes[gt_tag].shape[0])) - set(matched_indices))
        num_ignore_gt = get_ignores(gt_boxes[gt_tag][unmatched_indices], gt_boxes[~gt_tag, :4], bm_thresh)
        # compurte results
        eps = 1e-6
        k = len(matches)
        m = gt_tag.sum() - num_ignore_gt
        n = dt_boxes.shape[0] - num_ignore_dt
        ratio = k / (m + n -k + eps)
        recall = k / (m + eps)
        cover = k / (n + eps)
        noise = 1 - cover
        result_dict = dict(ratio = ratio, recall = recall, cover = cover,
            noise = noise, k = k, m = m, n = n)
        result_queue.put_nowait(result_dict)


def gather(results):
    assert len(results)
    img_num = 0
    for result in results:
        if result['n'] != 0 or result['m'] != 0:
            img_num += 1
    mean_ratio = np.sum([rb['ratio'] for rb in results]) / img_num
    mean_cover = np.sum([rb['cover'] for rb in results]) / img_num
    mean_recall = np.sum([rb['recall'] for rb in results]) / img_num
    mean_noise = 1 - mean_cover
    valids = np.sum([rb['k'] for rb in results])
    total = np.sum([rb['n'] for rb in results])
    gtn = np.sum([rb['m'] for rb in results])

    #line = 'mean_ratio:{:.4f}, mean_cover:{:.4f}, mean_recall:{:.4f}, mean_noise:{:.4f}, valids:{}, total:{}, gtn:{}'.format(
    #    mean_ratio, mean_cover, mean_recall, mean_noise, valids, total, gtn)
    line = 'mean_ratio:{:.4f}, valids:{}, total:{}, gtn:{}'.format(
        mean_ratio, valids, total, gtn)
    return line, mean_ratio


def common_process(func, cls_list, nr_procs=10):
    total = len(cls_list)
    stride = math.ceil(total / nr_procs)
    result_queue = Queue(10000)
    results, procs = [], []
    for i in range(nr_procs):
        start = i*stride
        end = np.min([start+stride,total])
        sample_data = cls_list[start:end]
        p = Process(target= func,args=(result_queue, sample_data))
        p.start()
        procs.append(p)
    for i in range(total):
        t = result_queue.get()
        if t is None:
            continue
        results.append(t)
    for p in procs:
        p.join()
    return results


# ---------------------------------- Basic functions ----------------------------------
def load_json_lines(fpath):
    print(fpath)
    assert os.path.exists(fpath)
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]
    return records


def save_json_lines(content,fpath):
    with open(fpath,'w') as fid:
        for db in content:
            line = json.dumps(db)+'\n'
            fid.write(line)


def load_bboxes(dict_input, key_name, key_box, key_score=None, key_tag=None):
    assert key_name in dict_input
    if len(dict_input[key_name]) < 1:
        return np.empty([0, 5])
    else:
        assert key_box in dict_input[key_name][0]
        if key_score:
            assert key_score in dict_input[key_name][0]
        if key_tag:
            assert key_tag in dict_input[key_name][0]
    if key_score:
        if key_tag:
            bboxes = np.vstack([np.hstack((rb[key_box], rb[key_score], rb[key_tag])) for rb in dict_input[key_name]])
        else:
            bboxes = np.vstack([np.hstack((rb[key_box], rb[key_score])) for rb in dict_input[key_name]])
    else:
        if key_tag:
            bboxes = np.vstack([np.hstack((rb[key_box], rb[key_tag])) for rb in dict_input[key_name]])
        else:
            bboxes = np.vstack([rb[key_box] for rb in dict_input[key_name]])
    return bboxes


def clip_boundary(boxes,height,width):
    assert boxes.shape[-1]>=4
    boxes[:,0] = np.minimum(np.maximum(boxes[:,0],0), width - 1)
    boxes[:,1] = np.minimum(np.maximum(boxes[:,1],0), height - 1)
    boxes[:,2] = np.maximum(np.minimum(boxes[:,2],width), 0)
    boxes[:,3] = np.maximum(np.minimum(boxes[:,3],height), 0)
    return boxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze a json result file with iou match')
    parser.add_argument('--detfile', required=True, help='path of json result file to load')
    parser.add_argument('--target_key', required=True)
    args = parser.parse_args()
    evaluation_all(args.detfile, args.target_key)