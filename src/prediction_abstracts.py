class Annotation:
    def __init__(self, box=None, score=0.0, label='', ind=-1):
        if box is None:
            box = []
        self.box = box
        self.score = max(score, 0.01)
        self.label = label
        self.ind = ind

    def to_str(self):
        bbox = '{}-{}-{}-{}-{}-{}-{}'.format(self.label, self.ind, self.score, self.box[0], self.box[1], self.box[2],
                                             self.box[3])
        return bbox


class Box:
    def __init__(self, arr):
        self.left = arr[0]
        self.top = arr[1]
        self.right = arr[2]
        self.bottom = arr[3]

    def area(self):
        return (self.right - self.left) * (self.bottom - self.top)


