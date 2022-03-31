import imp
from pyparsing import col
import torch
from torch import nn
import torch.utils.data
import PIL.Image as Image
import os
import numpy
import torchvision.transforms.functional
import torchvision
from tqdm import tqdm
class Voc2012(torch.utils.data.Dataset):
    def __init__(self, root, filename, image_size, transform, target_transform):
        super(Voc2012, self).__init__()
        "VOC2012 dataset different classifier of 21"
        self.voc_color = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                          [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                          [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                          [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                          [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                          [0, 64, 128]]
        self.voc_classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                            'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                            'diningtable', 'dog', 'horse', 'motorbike', 'person',
                            'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
        self.colormap = torch.zeros(256 ** 3, dtype=torch.long)
        for i, color in enumerate(self.voc_color):
            self.colormap[(color[0] * 256 + color[1]) * 256 + color[2]] = i

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.filename = filename
        self.image_size = image_size
        self.image_root = os.path.join(self.root, 'JPEGImages')
        self.segC_root = os.path.join(self.root, 'SegmentationClass')
        with open(os.path.join(self.root, self.filename), 'r') as f:
            self.image_filename = f.read().split()

        self._label()
        
    def __getitem__(self, item):
        pil_image_x = Image.open(os.path.join(self.image_root, self.image_filename[item] + '.jpg'))
        x_image = torch.tensor(numpy.transpose(numpy.array(pil_image_x), (2,0,1))).float()
        y_label = self.labels[item]

        x_image = self.pad(x_image, x_image.shape[1], x_image.shape[2], self.image_size)
        y_label = self.pad(y_label, y_label.shape[0], y_label.shape[1], self.image_size)

        rect = torchvision.transforms.RandomCrop.get_params(x_image, self.image_size)
        x_image = torchvision.transforms.functional.crop(x_image, *rect)
        y_label =  torchvision.transforms.functional.crop(y_label, *rect)

        return x_image / 255., y_label

    def __len__(self):
        return len(self.image_filename)

    def _label(self):
        labels = []
        for f in self.image_filename:
            pil_image_y = Image.open(os.path.join(self.segC_root, f + '.png')).convert('RGB')
            y_image = numpy.array(pil_image_y, dtype=numpy.int32)
            idx = (y_image[:,:,0] * 256 + y_image[:,:,1]) * 256 + y_image[:,:,2]
            labels.append(self.colormap[idx])
        self.labels = labels

#get subsuit size of image
    def pad(self, image, h, w, require_size):
        r_h, r_w = require_size[0:2]
        padding_h = max(r_h - h, 0)
        padding_w = max(r_w - w, 0)
        return torchvision.transforms.functional.pad(image, padding=[padding_h, padding_w])
        
def get_voc2012(require_data = 'train', transform = None, target_transform = None):
    return Voc2012('/VOCdevkit/VOC2012', 'ImageSets/Segmentation/' + require_data + '.txt', 
                        image_size=(256, 256), transform = transform, target_transform = target_transform)

# if __name__ == '__main__':
#     a = get_voc2012

