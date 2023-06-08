#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
# Thanks to YOLOX: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/tools/export_onnx.py

import argparse
import os
from loguru import logger
import sys
sys.path.append('..')

import torch
from torch import nn

from utils.misc import SiLU
from utils.misc import load_weight, replace_module

from config import  build_model_config
from models.detectors import build_model


def make_parser():
    parser = argparse.ArgumentParser("YOLO ONNXRuntime")
    # basic
    parser.add_argument('-size', '--img_size', default=640, type=int,
                        help='the max size of input image')
    parser.add_argument("--input", default="images", type=str,
                        help="input node name of onnx model")
    parser.add_argument("--output", default="output", type=str,
                        help="output node name of onnx model")
    parser.add_argument("-o", "--opset", default=11, type=int,
                        help="onnx opset version")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch size")
    parser.add_argument("--dynamic", action="store_true", default=False,
                        help="whether the input shape should be dynamic or not")
    parser.add_argument("--no-onnxsim", action="store_true", default=False,
                        help="use onnxsim or not")
    parser.add_argument("-f", "--exp_file", default=None, type=str,
                        help="experiment description file")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    parser.add_argument('--save_dir', default='../weights/onnx/', type=str,
                        help='Dir to save onnx file')

    # model
    parser.add_argument('-m', '--model', default='yolov1', type=str,
                        help='build yolo')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('-ct', '--conf_thresh', default=0.1, type=float,
                        help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.5, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=100, type=int,
                        help='topk candidates for testing')
    parser.add_argument('-nc', '--num_classes', default=80, type=int,
                        help='topk candidates for testing')
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False,
                        help='fuse Conv & BN')

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    device = torch.device('cpu')

    # Dataset & Model Config
    model_cfg = build_model_config(args)

    # build model
    model = build_model(args, model_cfg, device, args.num_classes, False, deploy=True)

    # replace nn.SiLU with SiLU
    model = replace_module(model, nn.SiLU, SiLU)

    # load trained weight
    model = load_weight(model, args.weight, args.fuse_conv_bn)
    model = model.to(device).eval()

    logger.info("loading checkpoint done.")
    dummy_input = torch.randn(args.batch_size, 3, args.img_size, args.img_size)

    # save onnx file
    save_path = os.path.join(args.save_dir, str(args.opset))
    os.makedirs(save_path, exist_ok=True)
    output_name = os.path.join(args.model + '.onnx')
    output_path = os.path.join(save_path, output_name)

    torch.onnx._export(
        model,
        dummy_input,
        output_path,
        input_names=[args.input],
        output_names=[output_name],
        dynamic_axes={args.input: {0: 'batch'},
                      output_name: {0: 'batch'}} if args.dynamic else None,
        opset_version=args.opset,
    )

    logger.info("generated onnx model named {}".format(output_path))

    if not args.no_onnxsim:
        import onnx

        from onnxsim import simplify

        input_shapes = {args.input: list(dummy_input.shape)} if args.dynamic else None

        # use onnxsimplify to reduce reduent model.
        onnx_model = onnx.load(output_path)
        model_simp, check = simplify(onnx_model,
                                     dynamic_input_shape=args.dynamic,
                                     input_shapes=input_shapes)
        assert check, "Simplified ONNX model could not be validated"

        # save onnxsim file
        save_path = os.path.join(save_path, 'onnxsim')
        os.makedirs(save_path, exist_ok=True)
        output_path = os.path.join(save_path, output_name)
        onnx.save(model_simp, output_path)
        logger.info("generated simplified onnx model named {}".format(output_path))


if __name__ == "__main__":
    main()
