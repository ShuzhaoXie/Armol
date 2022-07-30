import numpy as np
from src.common import safe_divide
from src.ablation import cal_IOU


# generate the reward based on the ground_truth
def compare_gt_m2(predict, gt, threshold=0.5):
    predict = sorted(predict, key=lambda s: s.score, reverse=True)
    vp = [False] * len(predict)
    vg = [False] * len(gt)
    for i, a1 in enumerate(predict):
        for j, a2 in enumerate(gt):
            if a1.ind == a2.ind and cal_IOU(a1, a2) >= threshold and (vg[j] == False):
                # print('in')
                vp[i] = True
                vg[j] = True

    len_p = len(predict)
    len_g = len(gt)
    tp = 0
    fp = 0
    pre = []
    rec = []
    pre.append(0.)
    rec.append(0.)

    for i in range(len(predict)):
        if vp[i] is True:
            tp += 1
        else:
            fp += 1
        pre.append(safe_divide(tp, tp + fp))
        rec.append(safe_divide(tp, len_g))

    pre.append(0.)
    rec.append(1.)

    mpre = np.array(pre)
    mrec = np.array(rec)

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # look for recall value changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap
