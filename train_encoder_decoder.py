from tqdm import tqdm

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.ops as tvops
from torch.utils.data import DataLoader
import shutil 
from utils import Drawer
import numpy as np
import cv2

import os
from datetime import datetime
import argparse
# from mynet import SeqDataset as Dataset
from mynet import MyDataset as Dataset
from mynet import EDModel
from utils import AverageMeter, SimpleLogger
from validation import evaluate_EDModel as evaluate
import box_finder

def box_cam_intersection_torch_roialign(cams, boxes):
    assert isinstance(boxes, list), "boxes need to be a list"
    assert cams.shape[-2:] == (7,7)
    output_size = (7,7)
    roi_ = tvops.roi_align(cams, boxes, output_size=output_size, spatial_scale=7)
    scores = torch.mean(roi_, dim=(-2,-1))
    max_vals, max_idxs = scores.max(dim=1)
    
    return max_vals, max_idxs

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


def _save_checkpoint(path, epoch, model, optimizer=None):
    arch = type(model).__name__
    state = {
        'arch': arch,
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }
    if optimizer is not None:
        state = {**state, 'optimizer': optimizer.state_dict()}

    filename = os.path.join(
        path,
        'checkpoint-epoch{}.pth'.format(epoch)
    )
    torch.save(state, filename, _use_new_zipfile_serialization=False)
    print("Saving checkpoint: {} ...".format(filename))
    

def main(config, resume):

    # parameters
    batch_size = config.get('batch_size', 32)
    start_epoch = config['epoch']['start']
    max_epoch = config['epoch']['max']
    lr = config.get('lr', 0.0005)
    use_conf = config.get('use_conf', False)
    use_recon = config.get('use_recon', None)
    use_pbox = config.get('use_pbox', None)

    ## path
    save_path = config['save_path']
    timestamp = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(save_path, timestamp)

    result_path = os.path.join(save_path, 'result')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    model_path = os.path.join(save_path, 'model')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    dest = shutil.copy('train_encoder_decoder.py', save_path) 
    print("save to: ", dest)

    ## cuda or cpu
    if config['n_gpu'] == 0 or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("using CPU")
    else:
        device = torch.device("cuda:0")
    
    ## dataloader
    dataset = Dataset(phase='train', do_augmentations=True, return_box=True)
    data_loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        num_workers=1,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        # **loader_kwargs,
    )

    # val_dataset = Dataset(phase='val', do_augmentations=False)
    # val_data_loader = DataLoader(
    #     val_dataset,
    #     batch_size=int(batch_size),
    #     num_workers=1,
    #     shuffle=True,
    #     drop_last=True,
    #     pin_memory=True,
    #     # **loader_kwargs,
    # )
    val_data_loader = None

    # ## few shot
    # do_few_shot = False
    # if do_few_shot:
    #     fs_dataset = Dataset(
    #         phase='train', 
    #         do_augmentations=False, 
    #         metafile_path='metadata/detection_train_images.json')
    #     fs_data_loader = DataLoader(
    #         fs_dataset,
    #         batch_size=int(128),
    #         num_workers=1,
    #         shuffle=True,
    #         pin_memory=True,
    #         # **loader_kwargs,
    #     )


    ## CNN model
    output_dim = 3
    model = EDModel(output_dim, resnet_type='resnet34')
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    model = model.to(device)
    model.train()
    print(model)

    ## loss
    criterion = nn.CrossEntropyLoss(reduction='none')

    ## optimizer
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optim_params={
        'lr': lr,
        'weight_decay': 0,
        'amsgrad': False,
    }
    optimizer = torch.optim.Adam(params, **optim_params)
    lr_params = {
        'milestones':[],
        'gamma':0.1,
    }
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,**lr_params)

    loss_avg = AverageMeter()
    recon_loss_avg = AverageMeter()
    class_loss_avg = AverageMeter()
    mask_loss_avg = AverageMeter()
    acc_avg = AverageMeter()
    logger = SimpleLogger(['train_loss', 'train_acc', 'val_loss', 'val_acc'])


    box_thresholds = [0.05, 0.10, 0.15]
    best_idx = 0

    ## loop
    for epoch in range(start_epoch, max_epoch):

        loss_avg.reset()
        recon_loss_avg.reset()
        class_loss_avg.reset()
        mask_loss_avg.reset()

        for batch_idx, batch in tqdm(
            enumerate(data_loader), 
            total=len(data_loader),
            ncols = 80,
            desc= f'training epoch {epoch}',
            ):
            data = batch[0].to(device)
            gt_lbls = batch[1].to(device)
            gt_gt_lbls = batch[2].to(device)
            image = batch[3].to(device)
            normalized_boxes = batch[-1][:,:,:-1].to(device)
            
            ## 
            out = model(data, gt_lbls) ## logits: [B, NC]; conf: [B, 1]
            obj_logits, obj_attns, obj_masks, alphas = out

            preds = torch.max(obj_logits, dim=-1)[1]
            weights = model.compute_entropy_weight(obj_logits)
            
            loss = 0.0

            ## classification loss
            class_loss = criterion(obj_logits, gt_lbls) ## [B, 1]
            if use_conf:
                class_loss = (class_loss * (weights**2) + (1-weights)**2)
            loss += class_loss.mean()
            class_loss_avg.update(class_loss.mean(), batch_size)


            ## reconstruction
            if isinstance(use_recon, int) and epoch >= use_recon:
                size = obj_masks.shape[2:]
                recon_loss = F.l1_loss(obj_masks, F.interpolate(image.to(torch.float)/255, size, mode='bilinear'), reduction='none')
                loss += recon_loss.mean()
                recon_loss_avg.update(recon_loss.mean(), batch_size)
                
            ## use pseudo box as supervision
            if isinstance(use_pbox, int) and epoch >= use_pbox:
                # batch_boxes_float = []
                # box_num= 50
                # for img in image:
                #     img_numpy = img.cpu().permute(1,2,0).numpy()
                #     # print("processing")
                #     boxes_float = box_finder.find_boxes(img_numpy, 224, box_num=box_num)
                #     batch_boxes_float.append(boxes_float.to(device))

                scores, classes = box_cam_intersection_torch_roialign(obj_attns.permute(1,0,2,3), list(normalized_boxes))
                # batch_boxes_float = torch.stack(batch_boxes_float)
                box_num = normalized_boxes.shape[1]
                scores = scores.reshape(batch_size, box_num)
                classes = classes.reshape(batch_size, box_num)

                if epoch < use_pbox+10:
                    score_thres = box_thresholds[0]
                elif epoch < use_pbox+20:
                    score_thres = box_thresholds[1]
                else:
                    score_thres = box_thresholds[2]

                batch_boxes_weights = (scores>score_thres).to(torch.float32) * (classes == gt_lbls.unsqueeze(1)).to(torch.float32)
                H, W = alphas.shape[2:]
                # print(normalized_boxes[0][batch_boxes_weights[0]>0])

                box_vote_mask = make_mask_region_pyt(normalized_boxes, batch_boxes_weights, H, W, device=device)
    
                mask_loss = F.mse_loss(alphas[:,0], box_vote_mask)
                loss += mask_loss
                mask_loss_avg.update(mask_loss, batch_size)

                ## debug
                if True and batch_idx == 0:
                    root = 'train_vis'
                    if not os.path.exists(root):
                        os.mkdir(root)
                    for idx, (bmask, img) in enumerate(zip(box_vote_mask, image)):
                        bmask = bmask*255
                        img_box = Drawer.draw_heatmap(bmask.cpu().to(torch.uint8).numpy(), img.cpu().permute(1,2,0).numpy())
                        gt = gt_lbls[idx]
                        cam = obj_attns[gt, idx].detach().cpu()
                        # print(cam)
                        max_val = torch.max(cam)
                        min_val = torch.min(cam)
                        cam = (cam - min_val) / (max_val - min_val) *255
                        img_cam = Drawer.draw_heatmap(cam.to(torch.uint8).numpy(), img.cpu().permute(1,2,0).numpy())
                        img_box_cam = np.concatenate([img_box, img_cam], axis=0)
                        cv2.imwrite(os.path.join(root, f"cam_{idx}.png"), img_box_cam)
                        # print(idx, " pause")
                        # input()

            ## record
            loss_avg.update(loss.item(), batch_size)
            positive = ((gt_lbls == preds) + (gt_gt_lbls>2)).sum()
            batch_acc = positive.to(torch.float)/batch_size
            acc_avg.update(batch_acc.item(), batch_size)

            ## run backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() ## update

        ## each epoch
        logger.update(loss_avg.avg, 'train_loss')
        logger.update(acc_avg.avg, 'train_acc')
        print("train loss: ", loss_avg.avg)
        print("recon loss: ", recon_loss_avg.avg)
        print("class loss: ", class_loss_avg.avg)
        print("mask loss: ", mask_loss_avg.avg)
        print("train acc: ", acc_avg.avg)

        if val_data_loader is not None:
            log = evaluate(model.eval(), val_data_loader, device, use_conf=use_conf)
            model.train()

            logger.update(log['loss'], 'val_loss')
            logger.update(log['acc'], 'val_acc')
            print("val loss: ", log['loss'])
            print("val acc: ", log['acc'])

            best_idx = logger.get_best('val_acc',best='max')

        if best_idx == epoch or epoch%10==0:
            print('save ckpt')
            ## save ckpt
            _save_checkpoint(model_path, epoch, model)

        lr_scheduler.step()
        print()
    
    ## save final model
    _save_checkpoint(model_path, epoch, model)
    



##
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        help='the size of each minibatch')
    parser.add_argument('--reconstruction', default=5, type=int,
                        help='epoch starting the reconstruction loss')
    parser.add_argument('--pseudo_box', default=5, type=int,
                        help='epoch starting the pseudo box loss')
    parser.add_argument('-g', '--n_gpu', default=None, type=int,
                        help='if given, override the numb')
    parser.add_argument('-e', '--epoch', default=100, type=int,
                        help='if given, override the numb')
    parser.add_argument('-s', '--save_path', default='saved', type=str,
                        help='path to save')

    # We allow a small number of cmd-line overrides for fast dev
    args = parser.parse_args()

    config = {}
    config['batch_size'] = args.batch_size
    config['use_recon']=args.reconstruction
    config['use_pbox']=args.pseudo_box
    config['n_gpu'] = args.n_gpu
    config['save_path'] = args.save_path
    config['epoch'] = {
        'start': 0,
        'max': args.epoch
    }
    config['lr'] = 0.0001
    config['use_conf']=True

    for k, v in config.items():
        print(k, v)

    main(config, args.resume)