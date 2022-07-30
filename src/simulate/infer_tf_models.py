import enum
import json
import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from tensorflow.python.ops.gen_array_ops import broadcast_to_eager_fallback

import os
import sys
sys.path.append('..')
from common import TRAIN_RANK, WORK_DIR, file_lines_to_list, json_load, TEST_RANK


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8), im_width, im_height


def get_keypoint_tuples(eval_config):
    """Return a tuple list of keypoint edges from the eval config.

    Args:
      eval_config: an eval config containing the keypoint edges

    Returns:
      a list of edge tuples, each in the format (start, end)
    """
    tuple_list = []
    kp_list = eval_config.keypoint_edge
    for edge in kp_list:
        tuple_list.append((edge.start, edge.end))
    return tuple_list

# MODELS = {'centernet_with_keypoints': 'centernet_resnet50_v1_fpn_512x512_kpts_coco17_tpu-8', 'centernet_without_keypoints': 'centernet_resnet50_v1_fpn_512x512_kpts_coco17_tpu-8'}
MODELS_SELECT = {
	0 : 'centernet_resnet50_v1_fpn_512x512_kpts_coco17_tpu-8', # 29.3
	1 : 'centernet_resnet50_v2_512x512_kpts_coco17_tpu-8', # 27.6
	2 : 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', # 22.2
 	3 : 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8', # 28.2
  4 : 'ssd_mobilenet_v2_320x320_coco17_tpu-8', # 20.2
	5 : 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8', # 29.3
	6 : 'ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8' # 29.1
	
}

CKPT_DIR = f'{WORK_DIR}/simulate/ckpt'

model_display_name = 'centernet_with_keypoints' # @param ['centernet_with_keypoints', 'centernet_without_keypoints']
model_name = MODELS_SELECT[4]

pipeline_config = os.path.join(f'{WORK_DIR}/simulate/models/research/object_detection/configs/tf2/',
                                model_name + '.config')

model_dir = os.path.join(CKPT_DIR, model_name, 'checkpoint/')

print('pipeline_config', pipeline_config)
print('model_dir', model_dir)
# 'models/research/object_detection/test_data/checkpoint/'

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(
      model_config=model_config, is_training=False)


    
# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(
      model=detection_model)
ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()

def get_model_detection_function(model):
  """Get a tf.function for detection."""

  @tf.function
  def detect_fn(image):
    """Detect objects in image."""

    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

  return detect_fn

detect_fn = get_model_detection_function(detection_model)


train_image_names = json_load(TRAIN_RANK)

test_image_names = json_load(TEST_RANK)
 

def parse_class2label(path):
    i = 2
    j = 3
    ind2label = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        l = len(lines)
        while j < l:
            ind = lines[i].strip().split(':')[-1]
            label = lines[j].strip().split(':')[-1]
            ind = int(ind.strip())
            label = label[2:-1]
            # print(ind, label)
            ind2label[ind] = label
            i += 5
            j += 5
    # print(len(ind2label))
    return ind2label
    


coco_names = parse_class2label(f'{WORK_DIR}/simulate/ms_coco_label.txt')

print(coco_names.keys())


ind2label = file_lines_to_list('coco.names')

label2ind = {}
for i, label in enumerate(ind2label):
    label2ind[label] = i


def write_results_to_file(file_path, classes, scores, boxes, width, height, min_score_thresh):
    f = open(file_path, 'w')
    for i, ind in enumerate(classes):
        score = scores[i]
        box = boxes[i]
        ind = int(ind+1)
        if score > min_score_thresh and ind in coco_names.keys():
            ymin, xmin, ymax, xmax = box
            (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
            label = coco_names[ind]
            real_ind = label2ind[label]
            bbox = '{}-{}-{}-{}-{}-{}-{}\n'.format(label, real_ind, float(score), int(left), int(top), int(right), int(bottom))
            print(bbox)
            f.write(bbox)
    f.close()
    
def infer(names, image_dir, res_dir, min_score_thresh=0.5):    
	for i, image_name in enumerate(names):
		try:
			image_path = os.path.join(image_dir, image_name)
			print('image_path', image_path)
			image_np, image_width, image_height = load_image_into_numpy_array(image_path)
			
			input_tensor = tf.convert_to_tensor(
			np.expand_dims(image_np, 0), dtype=tf.float32)
			detections, predictions_dict, shapes = detect_fn(input_tensor)
			# print(detections['detection_boxes'][0].numpy())
			# print(detections['detection_classes'][0].numpy())
			# print(detections['detection_scores'][0].numpy())
			# print(detections['num_detections'].numpy())
			# print('shape', shapes)
			write_results_to_file(os.path.join(res_dir, '{}.txt'.format(i)), 
							detections['detection_classes'][0].numpy(),
							detections['detection_scores'][0].numpy(),
							detections['detection_boxes'][0].numpy(), 
							image_width,
							image_height,
							min_score_thresh
							)
			print('{} image ok!'.format(i))
		except Exception:
			f1 = open(os.path.join(res_dir, '{}.txt'.format(i)), 'w')
			f1.close()

infer(test_image_names, f'{WORK_DIR}/data/test/images', f'{WORK_DIR}/predicted/model4/test')


		
# viz_utils.visualize_boxes_and_labels_on_image_array

