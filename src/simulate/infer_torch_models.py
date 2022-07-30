import random
import cv2
import json
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
# from google.colab.patches import cv2_imshow
import numpy as np
import torch
import torchvision
from common import *
import cv2 as cv

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries

# import some common detectron2 utilities

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = f'{WORK_DIR}/simulate/detectron_models/model_final_b275ba.pkl'
predictor = DefaultPredictor(cfg)

coco_names = file_lines_to_list('coco.names')

def write_results_to_file(file_path, classes, scores, boxes):
    f = open(file_path, 'w')
    for i, ind in enumerate(classes):
        score = scores[i]
        box = boxes[i]
        if score < 0.001:
            score = 0
        left, top, right, bottom = box
        ind = int(ind)
        label = coco_names[ind]
        bbox = '{}-{}-{}-{}-{}-{}-{}\n'.format(label, ind, float(
            score), int(left), int(top), int(right), int(bottom))
        print(bbox)
        f.write(bbox)
    f.close()


def infer(names, image_dir, res_dir, min_score_thresh=0.5):
    for i, image_name in enumerate(names):
        image_path = os.path.join(image_dir, image_name)
        print('image_path', image_path)
        im = cv.imread(image_path)
        outputs = predictor(im)
        # print(outputs["instances"].pred_classes)
        # print(outputs["instances"].pred_boxes)
        write_results_to_file(os.path.join(res_dir, '{}.txt'.format(i)),
                              to_numpy(outputs["instances"].pred_classes),
                              to_numpy(outputs["instances"].scores),
                              to_numpy(outputs["instances"].pred_boxes.tensor))
        print('{} image ok!'.format(i))



train_image_names = json_load(TRAIN_RANK)

test_image_names = json_load(TEST_RANK)

infer(test_image_names, TEST_IMAGE_DIR,
      f'{WORK_DIR}/predicted/model3/test')
