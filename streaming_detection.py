from tqdm import tqdm

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import scipy
import torchvision.ops as tvops
import skimage.morphology as skimg_morph
import cv2

import os
import argparse
from datetime import datetime
from mynet import StreamingDataloader 
from mynet import MyNet, EDModel, BoxOptimizer
from utils import Drawer, read_json
import box_finder


BOX_SCORE_THRESHOLD = 0.2

def box_cam_intersection(boxes, cams, scale_wh):
    scale_wh = (1, *scale_wh)
    cams_reshape = scipy.ndimage.zoom(cams, scale_wh) ## [3, 224, 398]
    boxes= np.array(boxes) ## [N, 4]
    def generate_mask(boxes, shape):
        shape = (boxes.shape[0], *shape)
        tmp = np.zeros(shape)
        for idx, box in enumerate(boxes):
            tmp[idx, box[1]:box[3], box[0]:box[2]] = 1
        return tmp

    H, W = cams_reshape.shape[1:]
    masks = generate_mask(boxes, (H, W)) ## [N, H, W] X [C, H, W]
    box_areas = (boxes[:,3]-boxes[:,1]) * (boxes[:,2]-boxes[:,0])
    scores = masks.reshape(-1,1,H,W) * cams_reshape.reshape(1,-1,H,W)
    scores = scores.sum(axis=(2,3)) / box_areas.reshape(-1,1)
    max_vals = np.amax(scores,axis=-1)
    max_idxs = np.argmax(scores,axis=-1)
    return max_vals, max_idxs


def box_cam_intersection_torch_roialign(cams, boxes):
    assert cams.shape[-2:] == (7,7)
    boxes = boxes.to(torch.float)
    output_size = (7,7)
    print(cams.shape)
    roi_ = tvops.roi_align(cams, [boxes], output_size=output_size, spatial_scale=7)
    # print("cams", cams.shape)
    print("roi", roi_.shape)
    box_areas = (boxes[:,3]-boxes[:,1]) * (boxes[:,2]-boxes[:,0])
    scores = torch.mean(roi_, dim=(-2,-1)) ##/ box_areas.reshape(-1,1) / output_size[0] / output_size[1]
    max_vals, max_idxs = scores.max(dim=1)
    # print(max_vals)
    # print(max_idxs)
    return max_vals, max_idxs


def rank_boxes(box_list, box_scores):
    assert len(box_list)==len(box_scores)
    data = zip(box_list, box_scores)
    sorted_by_second = sorted(data, key=lambda tup: tup[1], reverse = True)
    boxes, scores = list(zip(*sorted_by_second))
    
    return boxes, scores


class Detector(object):

    def __init__(self, model, data_loader, device):
        self.model = model
        self.data_loader = data_loader
        self.device = device

        self.selective_search_image_width = 448
        self.image_width = self.data_loader.imwidth

    def __call__(self, image_path, draw_path):
        with torch.no_grad():
            batch = self.data_loader(image_path)
            data = batch[0].to(self.device)
            image = batch[-1]

    

            ## run forward pass
            # out = self.model.detection(data.unsqueeze(0)) ## [B,N,H,W]
            # logits = self.model(data.unsqueeze(0))
            # preds = torch.max(logits, dim=-1)[1]
            # weights = self.model.compute_entropy_weight(logits)
            # print(preds, weights)

            out = self.model(data.unsqueeze(0), None)
            preds = torch.max(out[0], dim=-1)[1]

            if draw_path is not None:
                input()
                image_name = image_path.split('/')[-1]
                image_name = image_name.split('.')[0]
                filename = os.path.join(draw_path, f"detection_{image_name}")
                cams = out[1] ## [K, B, H, W]

                # ## normalize the cams    
                # max_val = torch.max(cams)
                # min_val = torch.min(cams)
                # cams = (cams - min_val) / (max_val - min_val)
                
                ## find boxes
                width, height = image.shape[1:]
                img_numpy = image.permute(1,2,0).numpy()
                boxes_float = box_finder.find_boxes(img_numpy, self.selective_search_image_width)
                # ratio = float(height)/width
                # image_size_for_ss_box = (self.selective_search_image_width, int(self.selective_search_image_width*ratio))
                # img_numpy = image.permute(1,2,0).numpy()
                # box_image_numpy = Drawer.resize_image(img_numpy, image_size_for_ss_box)
                # # boxes_float = ss_box_finder(box_image_numpy, normalized=True) ## np.array
                # boxes_float, _ = edge_box_finder(box_image_numpy, normalized=True)
                # boxes_float = torch.from_numpy(boxes_float).to(data.device)
                scores, classes = box_cam_intersection_torch_roialign(cams.permute(1,0,2,3), boxes_float)

                boxes = boxes_float.clone() ## COPY ONE
                boxes[:,0] = boxes[:,0]*width
                boxes[:,1] = boxes[:,1]*height
                boxes[:,2] = boxes[:,2]*width
                boxes[:,3] = boxes[:,3]*height
                boxes = boxes.numpy().astype(int)

                print(scores)
                print(classes)
                print(scores>BOX_SCORE_THRESHOLD)
        
                nocolor_ids = np.zeros((len(boxes),), dtype=int)-1
                Drawer.draw_boxes_on_image(boxes, img_numpy, nocolor_ids, filename+'_all_boxes', convert=True)
                Drawer.draw_boxes_on_image(boxes[scores>BOX_SCORE_THRESHOLD], img_numpy, classes[scores>BOX_SCORE_THRESHOLD], filename, convert=True)

                ## draw image
                if True:
                    labels = ['C', 'H', 'P']
                    cams = cams.permute(0,2,3,1)
                    heatmaps = Drawer.normalize_data_to_img255(cams.numpy())
                    for idx, heatmap in enumerate(heatmaps):
                        if idx == preds.item():
                            bool_inds = classes==idx
                            sorted_boxes, sorted_scores = rank_boxes(boxes[bool_inds], scores[bool_inds])
                            sorted_boxes = np.array(sorted_boxes)
                            sorted_scores = np.array(sorted_scores)
                            
                            cam_img_numpy = Drawer.draw_heatmap(heatmap, img_numpy, normalized=True)
                            cam_img_numpy = cam_img_numpy.astype(np.uint8)
                            # if len(sorted_boxes) > 10:
                            Drawer.draw_boxes_on_image(sorted_boxes[sorted_scores>BOX_SCORE_THRESHOLD], cam_img_numpy, classes[bool_inds], filename+labels[idx])
                            # else:
                            #     Drawer.draw_boxes_on_image(sorted_boxes, cam_img_numpy, classes[bool_inds], filename+labels[idx])



class NewDetector(object):
    def __init__(self, model, data_loader, device):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.box_optimizer = BoxOptimizer(batch_size=1, max_iter=30)


    def __call__(self, image_path, draw_path):
        with torch.no_grad():
            batch = self.data_loader(image_path)
            data = batch[0].to(self.device)
            image = batch[-1]
            H, W = data.shape[-2:]
            out = self.model(data.unsqueeze(0), None)
            obj_logits, _, obj_masks = out[:3]
            
            ## mask
            alphas = out[-1]
            pred = torch.max(obj_logits, dim=-1)[1]


        ## NUMPY DATA ON CPU
        if draw_path is not None:
            ## get cam
            ## alphas [1, 1, H0, W0] -> cam [1, H, W]  
            cam = F.interpolate(alphas, size=(H, W), mode='nearest').squeeze(0)

            ## normalize the cam    
            max_val = torch.max(cam)
            min_val = torch.min(cam)
            cam = (cam - min_val) / (max_val - min_val)

            ## convert to opencv data
            cam_numpy = cam.permute(1,2,0).cpu().numpy() ## [1, H, W] --> [H, W, 1] OPENCV DATA
            cam_numpy = np.uint8(cam_numpy*255)
            ret, cam_numpy = cv2.threshold(cam_numpy,128,255,cv2.THRESH_BINARY)

            ## convert to heatmap image
            for _ in range(5):
                cam_numpy = skimg_morph.binary_dilation(cam_numpy)
                cam_numpy = skimg_morph.binary_closing(cam_numpy)
                cam_numpy = skimg_morph.binary_erosion(cam_numpy)

            cam_numpy = np.uint8(cam_numpy*255)
            box = self.box_optimizer(cam_numpy.reshape(1, H, W))[0]
            box = box.detach().cpu()
            x = int(box[0,0]*W)
            y = int(box[0,1]*H)
            w = int(box[0,2]*W)
            h = int(box[0,3]*H)

            cam_numpy = cv2.applyColorMap(cam_numpy, cv2.COLORMAP_JET)
            cam_numpy = cv2.cvtColor(cam_numpy, cv2.COLOR_RGB2BGR)

            img_numpy = image.permute(1,2,0).numpy()

            label, path = image_path.split('/')[-2:]
            image_id = path.split('.')[0] ## str
            img = Drawer.draw_heatmap(cam_numpy, img_numpy)
            img = np.uint8(img)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2, cv2.LINE_AA)
            filename = os.path.join(draw_path, "test_{}_{}.png".format(image_id, pred.item()))
            cv2.imwrite(filename, img)
            



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
    data_loader = StreamingDataloader(imwidth=256)

    ## CNN model
    output_dim = 3
    # model = MyNet(output_dim)
    model = EDModel(output_dim)
    ## resume a ckpt
    checkpoint = torch.load(resume, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model = model.to(device)
    print(model)

    ## init a detector
    detector = NewDetector(model, data_loader, device)
    
    # ## perform detection
    # """
    # real-time feeding the image path to the detector
    # """
    # data_folder = '../test_detection/'
    # paths = [
    #     'H001.png',
    #     'H002.png',
    #     'H003.png',
    #     'H004.png',
    #     'H005.png'
    # ]
    
    test_path = './metadata/test_images.json'
    files = read_json(test_path)
    files.sort(key=lambda x: x[5:-4])
    image_paths = files

    for idx, p in tqdm(
        enumerate(image_paths), 
        total=len(image_paths),
        ncols = 80,
        desc= f'detection',
        ):
        detector(p, draw_path=save_path)
        
    

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