from tqdm import tqdm

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import skimage.morphology as skimg_morph

import os
import argparse
from datetime import datetime
from mynet import SeqDataset, MyDataset
# from mynet import MyNet
from mynet import EDModel 
from mynet import BoxOptimizer
from torch.utils.data import DataLoader
from utils import AverageMeter, Drawer
from iodine_like import RefineNetLSTM, SBD, IODINE

import cv2
import torchvision.ops as tvops


"""
batch_boxes: [B, N, 4]
batch_weights: [B, N]
batch_shape: B, H, W
"""
def make_mask_region_pyt(batch_normalized_boxes, batch_weights, H, W, device):
    batch_size = batch_normalized_boxes.shape[0]
    batch_canvas = []
    for b in range(batch_size):
        canvas = torch.zeros(H, W, device=device)
        x0 = (batch_normalized_boxes[b, :, 0]*W).to(torch.int)
        y0 = (batch_normalized_boxes[b, :, 1]*H).to(torch.int)
        x1 = (batch_normalized_boxes[b, :, 2]*W).to(torch.int)
        y1 = (batch_normalized_boxes[b, :, 3]*H).to(torch.int)
        w = batch_weights[b]
        for zipped in zip(x0, y0, x1, y1, w):
            canvas[zipped[1]:zipped[3], zipped[0]:zipped[2]] += zipped[4]

        ## normalize to 1.0
        max_val = canvas.max()
        min_val = canvas.min()
        canvas = (canvas - min_val) / (max_val - min_val + 10e-7)
        batch_canvas.append(canvas)

    return torch.stack(batch_canvas, dim=0)

def box_cam_intersection_torch_roialign(cams, boxes):
    assert isinstance(boxes, list), "boxes need to be a list"
    assert cams.shape[-2:] == (7,7)
    output_size = (7,7)
    roi_ = tvops.roi_align(cams, boxes, output_size=output_size, spatial_scale=7)
    scores = torch.mean(roi_, dim=(-2,-1))
    max_vals, max_idxs = scores.max(dim=1)
    
    return max_vals, max_idxs

def evaluate(model, data_loader, device, draw_path=None, use_conf=False):

    ## set model
    model.eval()
    model = model.to(device)

    fc_weights = model.head.weight.data
    # print(fc_weights) ## [N, C]
    # print(fc_weights.shape)
    # input()

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
            gt_lbls = batch[1].to(device)
            gt_gt_lbls = batch[2].to(device)
            images = batch[3]
            image_ids = batch[4]

            ## run forward pass
            batch_size = data.shape[0]
            out, feat = model.forward_eval(data) ## out: [B, N]; feat: [B, C, H, W] 
            preds = torch.max(out, dim=-1)[1]


            ## compute loss
            class_loss = criterion(out, gt_lbls) ## [B, 1]
            if use_conf:
                weights = model.compute_entropy_weight(out)
                loss = (class_loss * (weights**2) + (1-weights)**2).mean()
            else:
                loss = class_loss.mean()

            ## record
            loss_avg.update(loss.item(), batch_size)
            positive = ((gt_lbls == preds) + (gt_gt_lbls>2)).sum()
            batch_acc = positive.to(torch.float)/batch_size
            acc_avg.update(batch_acc.item(), batch_size)

            if draw_path is not None:
                ## get cam
                preds = torch.max(out, dim=-1)[1]
                B,C,H,W = feat.shape
                cam = fc_weights[preds,:].unsqueeze(-1) * feat.reshape(B, C, -1) ## [B, C] * [B, C, H, W]
                cam = torch.sum(cam, dim=1).reshape(B, H, W)
                ## normalize the cam    
                max_val = torch.max(cam)
                min_val = torch.min(cam)
                cam = (cam - min_val) / (max_val - min_val)
                ## convert to heatmap image
                cam_numpy = cam.permute(1,2,0).numpy()
                img_numpy = images[0].permute(1,2,0).numpy()
                filename = os.path.join(draw_path, f"test_{image_ids[0]}_{preds.item():d}_{weights.item():4.2f}")
                drawer.draw_heatmap(cam_numpy, img_numpy, filename)

    return {"loss": loss_avg.avg, "acc": acc_avg.avg}
    # print("test loss: ", loss_avg.avg)





def evaluate_EDModel(em_model, data_loader, device, draw_path=None, use_conf=False):

    ## set model
    em_model.eval()
    em_model = em_model.to(device)

    ## loss
    criterion = nn.CrossEntropyLoss(reduction='none')
    loss_avg = AverageMeter()
    acc_avg = AverageMeter()
    drawer = Drawer()

    box_optimizer = BoxOptimizer(batch_size=1, max_iter=30)


    for batch_idx, batch in tqdm(
        enumerate(data_loader), 
        total=len(data_loader),
        ncols = 80,
        desc= f'testing',
        ):

        # if batch_idx > 50:
        #     break

        with torch.no_grad():
            data = batch[0].to(device)
            gt_lbls = batch[1].to(device)
            gt_gt_lbls = batch[2].to(device)
            images = batch[3]
            image_ids = batch[4]
            normalized_boxes = batch[5][:,:,:4]

            ## run forward pass
            batch_size, _, H, W = data.shape
            out = em_model.forward(data, gt_lbls)                
            obj_logits, obj_attns, obj_masks = out[:3]
            alphas = out[-1]

            preds = torch.max(obj_logits, dim=-1)[1]

            ## compute loss
            class_loss = criterion(obj_logits, gt_lbls) ## [B, 1]
            if use_conf:
                weights = em_model.compute_entropy_weight(obj_logits)
                loss = (class_loss * (weights**2) + (1-weights)**2).mean()
            else:
                loss = class_loss.mean()

            ## record
            loss_avg.update(loss.item(), batch_size)
            positive = ((gt_lbls == preds) + (gt_gt_lbls>2)).sum()
            batch_acc = positive.to(torch.float)/batch_size
            acc_avg.update(batch_acc.item(), batch_size)

            scores, classes = box_cam_intersection_torch_roialign(obj_attns.permute(1,0,2,3), list(normalized_boxes))
            # batch_boxes_float = torch.stack(batch_boxes_float)
            box_num = normalized_boxes.shape[1]
            scores = scores.reshape(batch_size, box_num)
            classes = classes.reshape(batch_size, box_num)
            batch_boxes_weights = (scores>0.05).to(torch.float32) * (classes == gt_lbls.unsqueeze(1)).to(torch.float32)
            batch_boxes_weights = torch.zeros_like(batch_boxes_weights)+1.0
            H, W = alphas.shape[2:]
            # print(normalized_boxes[0][batch_boxes_weights[0]>0])

            box_vote_mask = make_mask_region_pyt(normalized_boxes, batch_boxes_weights, H, W, device=device)

        if draw_path is not None:
            ## get cam
            preds = torch.max(obj_logits, dim=-1)[1]
            # cam = alphas[:, 0]
            cam = obj_attns[preds.item()]
            cam = F.interpolate(cam.unsqueeze(0), size=(W, H), mode='bilinear').squeeze(0)

            ## normalize the cam    
            max_val = torch.max(cam)
            min_val = torch.min(cam)
            cam = (cam - min_val) / (max_val - min_val)

            ## convert to opencv data
            cam_numpy = cam.permute(1,2,0).cpu().numpy() ## [1, H, W] --> [H, W, 1] OPENCV DATA
            cam_numpy = np.uint8(cam_numpy*255)
            

            ## convert to region
            if False:
                ret,cam_numpy = cv2.threshold(cam_numpy,128,255,cv2.THRESH_BINARY)
                for _ in range(5):
                    cam_numpy = skimg_morph.binary_dilation(cam_numpy)
                    cam_numpy = skimg_morph.binary_closing(cam_numpy)
                    cam_numpy = skimg_morph.binary_erosion(cam_numpy)
                cam_numpy = np.uint8(cam_numpy*255)

                box = box_optimizer(cam_numpy.copy().reshape(1, H, W))[0]
                box = box.detach().cpu()
                x = int(box[0,0]*W)
                y = int(box[0,1]*H)
                w = int(box[0,2]*W)
                h = int(box[0,3]*H)

            cam_numpy = cv2.applyColorMap(cam_numpy, cv2.COLORMAP_JET)
            # cv2.imwrite("cam.png", cam_numpy)

            
            img_numpy = images[0].permute(1,2,0).numpy()
            img_numpy = cv2.resize(img_numpy, (H, W))

            cam_img_numpy = cam_numpy*0.3 + img_numpy*0.5
            cam_img_numpy = np.uint8(cam_img_numpy)
            cam_img_numpy = cv2.cvtColor(cam_img_numpy, cv2.COLOR_RGB2BGR)
            
            # cv2.rectangle(cam_img_numpy, (x, y), (x+w, y+h), (0, 0, 255), 2, cv2.LINE_AA)

            # recon_img_numpy = obj_masks[0].permute(1,2,0).cpu().numpy()
            # recon_img_numpy = recon_img_numpy*255
            recon_img_numpy = (box_vote_mask*255).reshape(H, W, 1) * 0.3 + img_numpy * 0.5
            recon_img_numpy = np.uint8(recon_img_numpy)

            filename = os.path.join(draw_path, f"test_{image_ids[0]}_{preds.item():d}_{weights.item():4.2f}")
            # drawer.draw_heatmap(cam_numpy, img_numpy, filename)
            drawer.draw_src_and_reconstructed_image(cam_img_numpy, recon_img_numpy, filename)

    

    return {"loss": loss_avg.avg, "acc": acc_avg.avg}
    # print("test loss: ", loss_avg.avg)






def main(resume, use_cuda=False, use_augment=False):
    
    ## path
    if True:
        timestamp = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join('test_result', timestamp)
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
    # test_path = './metadata/test_images.json'
    # new_test_path = './metadata/new_test_images.json'
    # dataset = SeqDataset(
    #     phase='test', 
    #     do_augmentations=False,
    #     metafile_path = test_path)

    dataset = MyDataset(
        phase='test', 
        do_augmentations=False, return_box=True)

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
    )


    ## CNN model
    output_dim = 3
    model = EDModel(output_dim)
    ## resume a ckpt
    checkpoint = torch.load(resume, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    # evaluate
    # log = evaluate(model, data_loader, device, draw_path=save_path, use_conf=True)
    log = evaluate_EDModel(model, data_loader, device, draw_path=save_path, use_conf=True)


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