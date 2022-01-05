import glob
import json
import os
import shutil

import numpy as np
import torch

from src.prediction_abstracts import Annotation, Box

def json_load(path):
    with open(path, 'r') as f:
        res = json.load(f)
    return res


def json_dump(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)
        
WORK_DIR = '/home/szxie/armor'

ONLINE = False

CUR_DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(CUR_DEVICE))

TRAIN_RANK = json_load(os.path.join(WORK_DIR, 'src/scripts/train_rank.json'))
TEST_RANK = json_load(os.path.join(WORK_DIR, 'src/scripts/test_rank.json'))
SYNONYM = json_load(os.path.join(WORK_DIR, 'src/scripts/word2num.json'))

ALPHA = -0.1

DATA_DIR = os.path.join(WORK_DIR, 'data')
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')
TRAIN_IMAGE_DIR = os.path.join(TRAIN_DATA_DIR, 'images')
TEST_IMAGE_DIR = os.path.join(TEST_DATA_DIR, 'images')

CLOUD2RES_DIR = {
    "aws": os.path.join(TEST_DATA_DIR, 'aws'),
    "azure": os.path.join(TEST_DATA_DIR, 'azure'),
    "google": os.path.join(TEST_DATA_DIR, 'google'),
    "ali": os.path.join(TEST_DATA_DIR, 'ali'),
    "gt": os.path.join(TEST_DATA_DIR, 'ground_truth'),
    "gt_train": os.path.join(TRAIN_DATA_DIR, 'ground_truth'),
    "aws_train": os.path.join(TRAIN_DATA_DIR, 'aws'),
    "azure_train": os.path.join(TRAIN_DATA_DIR, 'azure'),
    "google_train": os.path.join(TRAIN_DATA_DIR, 'google')
}


OPT_DIR = os.path.join(WORK_DIR, 'results')

# CLOUD2RES_DIR = {
#     "aws": os.path.join(PREDICTED_DIR, 'aws/test'),
#     "azure": os.path.join(PREDICTED_DIR, 'azure/test'),
#     "google": os.path.join(PREDICTED_DIR, 'google/test'),
#     "ali": os.path.join(PREDICTED_DIR, 'ali/test'),
#     "model_0": os.path.join(PREDICTED_DIR, 'model0/test'),
#     "model_1": os.path.join(PREDICTED_DIR, 'model1/test'),
#     "model_2": os.path.join(PREDICTED_DIR, 'model2/test'),
#     "model_3": os.path.join(PREDICTED_DIR, 'model3/test'),
#     "model_4": os.path.join(PREDICTED_DIR, 'model4/test'),
#     "model_5": os.path.join(PREDICTED_DIR, 'model5/test'),
#     "gt_train": '/home/szxie/dds/coco/gt_train',
#     "aws_train": os.path.join(PREDICTED_DIR, 'aws/train'),
#     "azure_train": os.path.join(PREDICTED_DIR, 'azure/train'),
#     "google_train": os.path.join(PREDICTED_DIR, 'google/train')
# }



class Detections:
    def __init__(self, cloud, rank):
        self.det = []
        for i, image_name in enumerate(rank):
            if cloud == 'gt':
                self.det.append(load_ground_truth_by_id(CLOUD2RES_DIR[cloud], i))
            else:
                self.det.append(load_res_by_id(CLOUD2RES_DIR[cloud], i))
        print('load {} detections'.format(cloud))

    def get(self, i):
        return self.det[i]
    
def to_numpy(var, dtype=np.float64, gpu_used=True):
    return var.cpu().data.numpy().astype(dtype) if gpu_used else var.data.numpy().astype(dtype)

def to_tensor(ndarray, gpu_0=CUR_DEVICE, gpu_used=True):
    if gpu_used:
        return torch.from_numpy(ndarray).to(device=gpu_0).type(torch.float32)
    else:
        return torch.from_numpy(ndarray).type(torch.float32)


def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def check_exist_dir_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def check_exist_file_path(path):
    if os.path.exists(path):
        os.remove(path)


def parse_day_hour(cap_name):
    day = cap_name.split('_')[0][6:8]
    hour = cap_name.split('_')[1][:2]
    return day, hour


def parse_hour(name):
    return name.split('_')[0]


def parse_day(name):
    return name.split('_')[-1]


def combine_day_hour(day, hour):
    return '{}_{}'.format(day, hour)


def path_join(p1, p2):
    return os.path.join(p1, p2)

# load ground truth of test images (OLD VERSION)
def load_ground_truth(gt_dir, image_path):
    image_name = image_path.split('/')[-1]
    path = path_join(gt_dir, image_name + '.txt')
    annotations = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split('-')
            label = data[0]
            score = 1
            box = list(map(lambda x: int(x), data[2:]))
            ind = SYNONYM[label]
            anno = Annotation(box, score, label, ind)
            annotations.append(anno)

    return annotations

# load ground truth (NEW VERSION)
def load_ground_truth_by_id(gt_dir, i):
    # image_name = image_path.split('/')[-1]
    # path = path_join(gt_dir, image_name + '.txt')
    file_path = os.path.join(gt_dir, '{}.txt'.format(i))
    annotations = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split('-')
            label = data[0]
            score = 1
            box = list(map(lambda x: int(x), data[2:]))
            ind = SYNONYM[label]
            anno = Annotation(box, score, label, ind)
            annotations.append(anno)

    return annotations

def load_res_by_id(dir_path, i):
    file_path = os.path.join(dir_path, '{}.txt'.format(i))
    try:
        results = []
        lines = file_lines_to_list(file_path)
        for line in lines:
            label, ind, score, left, top, right, bottom = line.split('-')
            label = label.lower()
            if SYNONYM.get(label) is None:
                continue
            score = float(score)
            box = list(map(int, [left, top, right, bottom]))
            anno = Annotation(box, score, label, int(ind))
            results.append(anno)
    except FileNotFoundError as e:
        # print(e)
        results = []

    return results

def safe_divide(a, b):
    if b == 0:
        return 0
    else:
        return a / b

def voc_ap(rec, prec):
    rec.insert(0, 0.0)  # insert 0.0 at beginning of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at beginning of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre

def cal_IOU(x, y):
    # left, top, right, bottom
    a = Box(x.box)
    b = Box(y.box)

    w = max(0, min(a.right, b.right) - max(a.left, b.left))
    h = max(0, min(a.bottom, b.bottom) - max(a.top, b.top))

    intersection = w * h
    union = a.area() + b.area() - intersection

    return intersection / union

def cal_mAP(predicted_dir_path, gt_dir=os.path.join(WORK_DIR, 'data/test/ground_truth'), min_overlap=0.5):
    tmp_files_path = "tmp_files"
    if not os.path.exists(tmp_files_path):  # if it doesn't exist already
        os.makedirs(tmp_files_path)

    # results_files_path = './map_results/combine/test1'
    # if os.path.exists(results_files_path):
    #     shutil.rmtree(results_files_path)
    # os.makedirs(results_files_path)

    ground_truth_files_list = glob.glob(gt_dir + '/*.txt')
    print('groud_truth_file_list', ground_truth_files_list[:5])
    ground_truth_files_list.sort()
    gt_counter_per_class = {}

    gt_bbox_map = {}

    for txt_file in ground_truth_files_list:
        file_id = txt_file.split(".txt", 1)[0]  # 这里的file_id是 ‘ground-truth/1’
        file_id = os.path.basename(os.path.normpath(file_id))  # 这里的file id则是被处理成了‘1’
        # check if there is a correspondent predicted objects file
        lines_list = file_lines_to_list(txt_file)
        # create ground-truth dictionary
        bounding_boxes = []
        is_difficult = False
        print('common.py txt_file:', txt_file)
        for line in lines_list:
            class_name, class_id, confidence_score, left, top, right, bottom = line.split('-')
            class_name = class_name.lower()
            class_name_num = SYNONYM[class_name]
            bbox = left + " " + top + " " + right + " " + bottom
            bounding_boxes.append({"class_name": class_name_num, "bbox": bbox, "used": False})
            # count that object
            if class_name_num in gt_counter_per_class:
                gt_counter_per_class[class_name_num] += 1
            else:
                # if class didn't exist yet
                gt_counter_per_class[class_name_num] = 1
        # dump bounding_boxes into a ".json" file
        gt_bbox_map[int(file_id)] = bounding_boxes
        # with open(tmp_files_path + "/" + file_id + "_ground_truth.json", 'w') as outfile:
        #     json.dump(bounding_boxes, outfile)

    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)

    predicted_files_list = glob.glob(predicted_dir_path + '/*.txt')
    predicted_files_list.sort()
    class_name_num2pd = {}

    for class_index, class_name in enumerate(gt_classes):
        class_name_num = class_name
        bounding_boxes = []
        for txt_file in predicted_files_list:
            # print(txt_file)
            # the first time it checks if all the corresponding ground-truth files exist
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            lines = file_lines_to_list(txt_file)
            # print(file_id)
            for line in lines:
                # print(line)
                tmp_class_name, tmp_class_name_num, confidence, left, top, right, bottom = line.split('-')
                tmp_class_name_num = int(tmp_class_name_num)
                # tmp_class_name, confidence, left, top, right, bottom = line.split('-')
                # tmp_class_name = tmp_class_name.lower()
                # if tmp_class_name in SYNONYM:
                #     tmp_class_name_num = SYNONYM[tmp_class_name]
                # else:
                #     tmp_class_name_num = -1
                if tmp_class_name_num == class_name_num:
                    bbox = left + " " + top + " " + right + " " + bottom
                    bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})

        bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)

        class_name_num2pd[class_name_num] = bounding_boxes

        with open(tmp_files_path + "/" + str(class_name_num) + "_predictions.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    sum_AP = 0.0
    ap_dictionary = {}
    # with open(results_files_path + "/results.txt", 'w') as results_file:
    #     results_file.write("# AP and precision/recall per class\n")
    count_true_positives = {}
    print('[', end='')
    for class_index, class_name_num in enumerate(gt_classes):
        count_true_positives[class_name_num] = 0
        """
        Load predictions of that class
        """
        # predictions_file = tmp_files_path + "/" + str(class_name_num) + "_predictions.json"
        # predictions_data = json.load(open(predictions_file))

        predictions_data = class_name_num2pd[class_name_num]
        """
         Assign predictions to ground truth objects
        """
        nd = len(predictions_data)
        tp = [0] * nd  # creates an array of zeros of size nd
        fp = [0] * nd
        for idx, prediction in enumerate(predictions_data):
            file_id = prediction["file_id"]
            # gt_file = tmp_files_path + "/" + file_id + "_ground_truth.json"
            # ground_truth_data = json.load(open(gt_file))
            ground_truth_data = gt_bbox_map[int(file_id)]
            ovmax = -1
            gt_match = -1
            # load prediction bounding-box
            bb = [float(x) for x in prediction["bbox"].split()]
            for obj in ground_truth_data:
                # look for a class_name match
                if obj["class_name"] == class_name_num:
                    bbgt = [float(x) for x in obj["bbox"].split()]
                    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) \
                             + (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj

            # set minimum overlap
            # min_overlap = MINOVERLAP
            if ovmax >= min_overlap:
                if "difficult" not in gt_match:
                    if not bool(gt_match["used"]):
                        # true positive
                        tp[idx] = 1
                        gt_match["used"] = True
                        count_true_positives[class_name_num] += 1
                        # update the ".json" file
                        gt_bbox_map[int(file_id)] = ground_truth_data
                        # with open(gt_file, 'w') as f:
                        #     f.write(json.dumps(ground_truth_data))
                    else:
                        # false positive (multiple detection)
                        fp[idx] = 1
            else:
                # false positive
                fp[idx] = 1
            if ovmax > 0:
                status = "INSUFFICIENT OVERLAP"

        # print(tp)
        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        # print(tp)
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name_num]
        # print(rec)
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        # print(prec)

        ap, mrec, mprec = voc_ap(rec, prec)
        sum_AP += ap
        text = "{0:.2f}%".format(ap * 100) + " = " + str(
            class_name_num) + " AP  "  # class_name + " AP = {0:.2f}%".format(ap*100)
        """
        Write to results.txt
        """
        rounded_prec = ['%.2f' % elem for elem in prec]
        rounded_rec = ['%.2f' % elem for elem in rec]
        # results_file.write(
        #     text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")
        # print top_10_num
        # top_10_num = [0,2,56,73,39,41,60,9,45,26]
        # if class_name_num in top_10_num:
        #     print(ap, end=',')
        
        ap_dictionary[class_name_num] = ap
    # results_file.write("\n# mAP of all classes\n")
    mAP = sum_AP / n_classes
    print(']')
    text = "mAP = {0:.2f}%".format(mAP * 100)
    return mAP


def get_top_k(action, K = 1):
    res = []
    discrete_actions = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1]
    ]
    discrete_actions = to_tensor(np.array(discrete_actions))

    l2_dis = (discrete_actions - action).square().sum(axis=1).mul_(-1)

    topk_prob, topk_catid = torch.topk(l2_dis, K)

    ret_actions = discrete_actions[topk_catid]

    ret_actions = sorted(ret_actions, key=lambda s: torch.sum(s), reverse=False)

    ret_actions = torch.cat(ret_actions).reshape((K, 3))
    
    return ret_actions

def write_results_to_file(results, file_path):
    f = open(file_path, 'w')
    for r in results:
        bbox = '{}-{}-{}-{}-{}-{}-{}\n'.format(r.label, r.ind, r.score, int(r.box[0]), int(r.box[1]), int(r.box[2]), int(r.box[3]))
        f.write(bbox)
    f.close()
