import numpy as np
import os
import glob
import torch
import numpy

from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

class StreamingDataloader(object):
    classes_str = ['C', 'H', 'P', 'CP', 'PH', 'HC']
    classes = {'C': [0], 'H': [1], 'P': [2], 'CP': [0,2], 'PH': [1,2], 'HC': [0,1]}
    torch.manual_seed(0)

    def __init__(self, imwidth=224):
        ## parameters
        self.imwidth = imwidth
        
        ## data pre-processing
        self.normalize = transforms.Normalize(mean=[0.5084, 0.4224, 0.3769],
                                         std=[0.2599, 0.2371, 0.2323])
        self.initial_transforms = transforms.Compose([transforms.Resize((self.imwidth, self.imwidth))])
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image_path):
        image = Image.open(image_path)
        orig_image = TF.pil_to_tensor(image)
        image = self.initial_transforms(image.convert("RGB")) ## to PIL.rgb
        image = TF.pil_to_tensor(image) ## save the image tensor for visualization
        data = self.normalize(self.to_tensor(TF.to_pil_image(image)))
        return data, image, orig_image
