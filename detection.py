from tqdm import tqdm

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np


import os
import argparse
from datetime import datetime
from mynet import SeqDataset 
from mynet import MyNet
from torch.utils.data import DataLoader
from utils import AverageMeter, Drawer

def evaluate(model, data_loader, device, draw_path=None, use_conf=False):

    ## set model
    model.eval()
    model = model.to(device)

    fc_weights = model.head.weight.data

    ## loss
    criterion = nn.CrossEntropyLoss(reduction='none')
    loss_avg = AverageMeter()
    acc_avg = AverageMeter()
    drawer = Drawer()

    with torch.no_grad():
        for batch_idx, batch in tqdm(
            enumerate(data_loader), 
            total=len(data_loader),
            ncols = 80,
            desc= f'testing',
            ):

            data = batch[0].to(device)
            images = batch[-2]
            image_ids = batch[-1]

            ## run forward pass
            batch_size = data.shape[0]
            out, feat = model.forward_eval(data) ## out: [B, N]; feat: [B, C, H, W] 

            if draw_path is not None:
                ## get cam
                B,C,H,W = feat.shape
                N = fc_weights.shape[0]
                cam = fc_weights.unsqueeze(-1) * feat.reshape(B, C, -1) ## [N, C, 1] * [B, C, HW]
                cam = torch.sum(cam, dim=1).reshape(N, H, W)
                ## normalize the cam    
                max_val = torch.max(cam)
                min_val = torch.min(cam)
                cam = (cam - min_val) / (max_val - min_val)
                ## convert to heatmap image
                cam_numpy = cam.permute(1,2,0).numpy()
                img_numpy = images[0].permute(1,2,0).numpy()
                for idx in range(0,N):
                    filename = os.path.join(draw_path, f"test_{image_ids[0]}_{idx:d}")
                    drawer.draw_heatmap(filename, cam_numpy[:,:,idx], img_numpy)


def main(resume, use_cuda=False, use_augment=False):
    
    ## path
    if True:
        timestamp = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join('detection', timestamp)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print('make a test result folder: ', save_path)
    else:
        save_path = None


    ## cuda or cpu
    if use_cuda:
        device = torch.device("cuda:0")
        print("using cuda")
    else:
        device = torch.device("cpu")
        print("using CPU")

    if use_augment: 
        print("data are augmented randomly")


    ## dataloader
    test_path = './metadata/test_images.json'
    new_test_path = './metadata/new_test_images.json'
    detection_path = './metadata/detection_images.json'

    dataset = SeqDataset(
        phase='test', 
        do_augmentations=use_augment,
        metafile_path = detection_path,
        return_gt_label=False)

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
    )


    ## CNN model
    output_dim = 3
    model = MyNet(output_dim)
    ## resume a ckpt
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])


    ## evaluate
    log = evaluate(model, data_loader, device, draw_path=save_path, use_conf=True)
    print("val loss: ", log['loss'])
    print("val acc: ", log['acc'])



## main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--augment', action='store_true')

    args = parser.parse_args()

    assert args.resume is not None, "provide ckpt path to try again"
    main(args.resume, use_cuda=args.cuda, use_augment=args.augment)