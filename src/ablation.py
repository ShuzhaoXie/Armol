import functools

from src.prediction_abstracts import Annotation, Box


def c1(self, other):
    if self.ind < other.ind:
        return 1
    elif self.ind == other.ind:
        return 0
    else:
        return -1


def c2(self, other):
    if self.score < other.score:
        return 1
    elif self.score == other.score:
        return 0
    else:
        return -1

def cal_IOU(x, y):
    # left, top, right, bottom
    a = Box(x.box)
    b = Box(y.box)

    w = max(0, min(a.right, b.right) - max(a.left, b.left))
    h = max(0, min(a.bottom, b.bottom) - max(a.top, b.top))

    intersection = w * h
    union = a.area() + b.area() - intersection

    return intersection / union

def combine_blocks(block, threshold):
    sorted_annotations = sorted(block, key=functools.cmp_to_key(c2))
    # for anno in sorted_annotations:
    #     print('anno', anno.to_string())

    l_sorted_annotations = len(sorted_annotations)

    vis = [False] * l_sorted_annotations

    res = []

    for i, x in enumerate(sorted_annotations):
        if not vis[i]:
            res.append(x)
            vis[i] = True
            for j in range(i + 1, l_sorted_annotations):
                if not vis[j]:
                    y = sorted_annotations[j]
                    ratio = cal_IOU(x, y)
                    if ratio >= threshold:
                        vis[j] = True

    return res

def split_blocks(annotations, split_threshold=0.5):
    """
    split annotations in to blocks with only one category.
    :param split_threshold: to identify the same box
    :param annotations: annotations in one image
    :return: already-splited annotations
    """
    sorted_annotations = sorted(annotations, key=functools.cmp_to_key(c1))
    blocks = []
    len_annotations = len(annotations)
    vis = [False for _ in range(len_annotations)]
    for i in range(len_annotations):
        if not vis[i]:
            block = []
            a = sorted_annotations[i]
            block.append(a)
            vis[i] = True
            for j in range(i + 1, len_annotations):
                if not vis[j]:
                    b = sorted_annotations[j]
                    if a.ind == b.ind and cal_IOU(a, b) > split_threshold:
                        block.append(b)
                        vis[j] = True
            blocks.append(block)

    return blocks

def nms(results, nms_threshold=0.5):
    res = []
    blocks = split_blocks(results)
    for block in blocks:
        block = combine_blocks(block, threshold=nms_threshold)
        res.extend(block)
    return res


def wbf(results):
    res = []
    blocks = split_blocks(results)
    for block in blocks:
        sum_c = 0
        T = len(block)
        for i, a in enumerate(block):
            sum_c += a.score
        left = 0
        right = 0
        top = 0
        bottom = 0
        for i, a in enumerate(block):
            left += a.box[0] * a.score
            top += a.box[1] * a.score
            right += a.box[2] * a.score
            bottom += a.box[3] * a.score
        box = list(map(lambda x: int(x / sum_c), [left, top, right, bottom]))
        ave_score = sum_c / T
        # print(box)
        # print(ave_score)
        final_anno = Annotation(box=box, ind=block[0].ind, score=ave_score, label=block[0].label)
        res.append(final_anno)
        # refine confidence不一定需要呀
    return res


def soft_nms(results, soft_nms_threshold=0.5):
    res = []
    l_results = len(results)
    vis = [False] * l_results

    def check_vis():
        for v in vis:
            if not v:
                return False
        return True

    while check_vis() is False:
        m = 0
        mi = 0
        for i, r in enumerate(results):
            if m < r.score and not vis[i]:
                m = r.score
                mi = i
        vis[mi] = True
        x = results[mi]
        res.append(x)
        for i, r in enumerate(results):
            if not vis[i]:
                iou = cal_IOU(x, results[i])
                if iou > soft_nms_threshold:
                    results[i].score = max(results[i].score * (1 - iou), 0.01)
                    if results[i].score < soft_nms_threshold:
                        vis[i] = True

    return res

