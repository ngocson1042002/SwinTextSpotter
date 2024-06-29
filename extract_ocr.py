# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import Predictor
from detectron2.utils.visualizer_vintext import decoder
import numpy as np
import json
import h5py

def ctc_decode_recognition(rec):
    CTLABELS = [" ", "!", '"', "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";",
                "<", "=", ">", "?", "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
                "X", "Y", "Z", "[", "\\", "]", "^", "_", "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
                "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "ˋ", "ˊ", "﹒", "ˀ", "˜", "ˇ", "ˆ", "˒", "‑"]
    # ctc decoding
    last_char = False
    s = ''
    for c in rec:
        c = int(c)
        if 0<c < 107:
            s += CTLABELS[c-1]
            last_char = c
        elif c == 0:
            s += u''
        else:
            last_char = False
    if len(s) == 0:
        s = ' '
    s = decoder(s)

    return s


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)

    # -----
    from projects.SWINTS.swints import add_SWINTS_config
    add_SWINTS_config(cfg)
    # -----

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    predictor = Predictor(cfg)
    
    paths = sorted(glob.glob(os.path.join(args.input[0], '*.jpg')))
    if len(paths) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"

    for path in tqdm.tqdm(paths, disable=not args.output):
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="RGB")
        start_time = time.time()
        predictions = predictor.run_on_image(img, args.confidence_threshold, path)
        instances = predictions["instances"]

        height, width = instances.image_size
        
        rec = instances.pred_rec
        detected_texts = [ctc_decode_recognition(rrec) for rrec in rec]
        
        boxes = instances.pred_boxes.tensor.detach().cpu().numpy()
        boxes = boxes.tolist()

        scores = instances.scores.tolist()

        det_features = instances.det_features.detach().cpu().numpy()
        rec_features = instances.rec_features.detach().cpu().numpy()
        
        image_id = int(os.path.basename(path).split(".")[0])
        
        features = {
            "image_id": image_id,
            "det_features": det_features,
            "rec_features": rec_features,
            "bboxes": {i:{j:c for j, c in enumerate(box)} for i, box in enumerate(boxes)},
            "rec": {i: rec for i, rec in enumerate(detected_texts)},
            "score": {i: rec for i, rec in enumerate(scores)},
            "height": height,
            "width": width
        }
        np.save(f"/kaggle/working/ocr_features/{image_id}.npy", features)
#     with open(os.path.join(args.output, "ocr_results.json"),"w", encoding='utf-8') as jsonfile:
#         json.dump(results, jsonfile,ensure_ascii=False)
