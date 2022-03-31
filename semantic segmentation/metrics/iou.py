import numpy as np
import json
class Accumulator():
    def __init__(self):
        pass
    def add(self, *kwargs):
        pass
    def json(self):
        pass
    def __str__(self) -> str:
        pass
class SegClassIouAccumulator(Accumulator):
    def __init__(self, num_classes):
        super(SegClassIouAccumulator, self).__init__()
        self.num_classes = num_classes
        self.matrixs = np.zeros((self.num_classes, self.num_classes), dtype = np.int32)
#bincount only can deal with one shape and we should flatten then reshape
    def update_matrix(self, gt, dt):
        self.matrixs += np.bincount(self.num_classes * gt + dt, 
                                    minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
    
    def add(self, gt, dt):
        assert dt.dtype == np.uint32 and gt.dtype == np.uint32
        for g, d in zip(gt, dt):
            self.update_matrix(g.flatten(), d.flatten())
    def __str__(self):
        return json.dumps(self.json())
    def json(self):
        acc = np.diag(self.matrixs).sum() / self.matrixs.sum()
        class_acc = np.diag(self.matrixs) / self.matrixs.sum(axis = 1)

        class_iou = np.diag(self.matrixs) / (self.matrixs.sum(axis=0) + self.matrixs.sum(axis=1) - np.diag(self.matrixs))
        iou = np.nanmean(class_iou)
        #item and tolist is to make var into python var
        return {'acc':acc.item(0),'class_acc':class_acc.tolist(), 'class_iou':class_iou.tolist(), 'iou':iou.item(0)}

    def reset(self):
        self.matrixs[::] = 0
if __name__ == '__main__':
    accumulator = SegClassIouAccumulator(num_classes=3)
    dt = np.array([[1, 2], [0, 2]], dtype=np.uint32).reshape((1, 2, 2))
    gt = np.array([[1, 2], [0, 2]], dtype=np.uint32).reshape((1, 2, 2))
    accumulator.add(dt, gt)
    print(accumulator)
