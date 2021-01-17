import numpy as np
import os,sys,inspect
import glob
import torch
import numpy

from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from io import BytesIO
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from utils import *

import warnings
warnings.filterwarnings("ignore")

class JPEGNoise(object):
    def __init__(self, low=30, high=99):
        self.low = low
        self.high = high


    def __call__(self, im):
        H = im.height
        W = im.width
        rW = max(int(0.8 * W), int(W * (1 + 0.5 * torch.randn([]))))
        im = TF.resize(im, (rW, rW))
        buf = BytesIO()
        im.save(buf, format='JPEG', quality=torch.randint(self.low, self.high,
                                                          []).item())
        im = Image.open(buf)
        im = TF.resize(im, (H, W))
        return im


class PcaAug(object):
    _eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
    _eigvec = torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def __call__(self, im):
        alpha = torch.randn(3) * self.alpha
        rgb = (self._eigvec * alpha.expand(3, 3) * self._eigval.expand(3, 3)).sum(1)
        return im + rgb.reshape(3, 1, 1)

class SeqDataset(Dataset):
    phases = ['train', 'val', 'test']
    classes_str = ['C', 'H', 'P', 'CP', 'PH', 'HC']
    classes = {'C': [0], 'H': [1], 'P': [2], 'CP': [0,2], 'PH': [1,2], 'HC': [0,1]}
    torch.manual_seed(0)

    def __init__(
        self, phase='train', imwidth=224, 
        metafile_path = None,
        do_augmentations=False, img_ext='.png',
        return_gt_label=True,
        return_box=False
        ):
        
        ## parameters
        assert phase in self.phases
        self.imwidth = imwidth
        self.phase = phase
        self.train = True if phase != 'test' else False
        self.do_augmentations = do_augmentations
        self.return_gt_label = return_gt_label
        self.return_box = return_box

        ## read image paths
        if metafile_path is None:
            metafile_path = f'./metadata/{phase}_images.json'

        files = read_json(metafile_path)
        files.sort(key=lambda x: x[5:-4])
        self.image_paths = []

        if return_gt_label:
            for fname in files:
                if fname.split('/')[-2] in self.classes:
                    self.image_paths.append(fname)
        else:
            self.image_paths = files
        print("data set size: ", len(self.image_paths))

        
        if self.return_box:
            box_paths = read_json(f'./metadata/edge_box_{phase}.json')
            self.box_dict = {}
            for bp in box_paths:
                label, path = bp.split('/')[-2:]
                image_id = path.split('.')[0][:-4] ## str
                self.box_dict[image_id] = bp


        ## data pre-processing
        self.normalize = transforms.Normalize(mean=[0.5084, 0.4224, 0.3769],
                                         std=[0.2599, 0.2371, 0.2323])
        self.initial_transforms = transforms.Compose([transforms.Resize((self.imwidth, self.imwidth))])
        self.augmentations = transforms.RandomApply(
                    [
                        transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=0.01),
                        # transforms.RandomAffine(degrees=45,translate=(0.1,0.1),scale=(0.9,1.2))
                    ], 
                    p=0.3)
        self.to_tensor = transforms.ToTensor()
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = self.initial_transforms(image.convert("RGB")) ## to PIL.rgb
        if self.do_augmentations:
            image = self.augmentations(image)
        image = TF.pil_to_tensor(image) ## save the image tensor for visualization
        data = self.normalize(self.to_tensor(TF.to_pil_image(image)))

        label, path = image_path.split('/')[-2:]
        image_id = path.split('.')[0] ## str
        
        
        if self.return_gt_label and self.return_box:
            assert label in self.classes
            n = len(self.classes[label])
            rand_idx = torch.randperm(n)[0]
            target = self.classes[label][rand_idx]

            box_path = self.box_dict[image_id]
            boxes = read_json(box_path)[image_id]
            boxes = torch.tensor(boxes)
            boxes_float = boxes[:50, :] ## 50/200
            
            return data, target, self.classes_str.index(label), image, image_id, boxes_float
        
        elif self.return_gt_label:
            assert label in self.classes
            n = len(self.classes[label])
            rand_idx = torch.randperm(n)[0]
            target = self.classes[label][rand_idx]
            return data, target, self.classes_str.index(label), image, image_id
            
        else:
            return data, image, image_id


class MyDataset(Dataset):
    phases = ['train', 'val', 'test']
    classes_str = ['C', 'H', 'P', 'O', 'CP', 'PH', 'HC', ]
    classes = {'C': [0], 'H': [1], 'P': [2], 'O':[0,1,2], 'CP': [0,2], 'PH': [1,2], 'HC': [0,1]}

    torch.manual_seed(0)

    def __init__(
        self, phase='train', imwidth=224, 
        metafile_path = None,
        do_augmentations=False, img_ext='.png',
        return_gt_label=True,
        return_box=False
        ):
        
        ## parameters
        assert phase in self.phases
        self.imwidth = imwidth
        self.phase = phase
        self.train = True if phase != 'test' else False
        self.do_augmentations = do_augmentations
        self.return_gt_label = return_gt_label
        self.return_box = return_box

        ## read image paths
        if metafile_path is None:
            metafile_path = f'./metadata/ar1440_frame_labels.json'
            # metafile_path = f'./metadata/d435_frame_labels.json'
            frame_labels = read_json(metafile_path)

        # self.image_paths = list(frame_labels)
        self.image_paths = list(frame_labels.keys())
        self.labels = list(frame_labels.values())
        print("data set size: ", len(self.image_paths))
        if self.return_box:
            self.box_dict = read_json(f'./metadata/ar0144_640x480_20200115_box.json')

        ## data pre-processing
        self.normalize = transforms.Normalize(mean=[0.5084, 0.4224, 0.3769],
                                         std=[0.2599, 0.2371, 0.2323])
        self.initial_transforms = transforms.Compose([transforms.Resize((self.imwidth, self.imwidth))])
        self.augmentations = transforms.RandomApply(
                    [
                        transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=0.01),
                        # transforms.RandomAffine(degrees=45,translate=(0.1,0.1),scale=(0.9,1.2))
                    ], 
                    p=0.3)
        self.to_tensor = transforms.ToTensor()
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index] + '.png'
        image = Image.open(image_path)
        image = self.initial_transforms(image.convert("RGB")) ## to PIL.rgb
        if self.do_augmentations:
            image = self.augmentations(image)
        image = TF.pil_to_tensor(image) ## save the image tensor for visualization
        data = self.normalize(self.to_tensor(TF.to_pil_image(image)))

        path = image_path.split('/')[-1]
        image_id = path.split('.')[0] ## str
        label = self.labels[index]

        if len(set(label)) > 1:
            if set(label) == {'C', 'P'}:
                label = 'CP'
            elif set(label) == {'P', 'H'}:
                label = 'PH'
            elif set(label) == {'H', 'C'}:
                label = 'HC'
            else:
                print(label)
                raise NotImplementedError
        else:
            label = label[0]
            


        if self.return_gt_label and self.return_box:
            assert label in self.classes
            n = len(self.classes[label])
            rand_idx = torch.randperm(n)[0]
            target = self.classes[label][rand_idx]

            fname = "/home/yangle/codes/WSOL/20210115/ar0144_640x480_20200115_box/"
            box_path = os.path.join(fname, image_id+"_box.json")
            boxes = read_json(box_path)[image_id]
            boxes = torch.tensor(boxes)
            boxes_float = boxes[:50, :] ## 50/200
            
            return data, target, self.classes_str.index(label), image, image_id, boxes_float
        
        elif self.return_gt_label:
            assert label in self.classes
            n = len(self.classes[label])
            rand_idx = torch.randperm(n)[0]
            target = self.classes[label][rand_idx]
            return data, target, self.classes_str.index(label), image, image_id
            
        else:
            return data, image, image_id


if __name__ == '__main__':
    dataset = SeqDataset(phase='test')
    files = dataset.image_paths
    files.sort(key=lambda x: x[5:-4])
    print(files)
