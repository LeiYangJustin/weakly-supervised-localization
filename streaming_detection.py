from tqdm import tqdm

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import scipy
import torchvision.ops as tvops
import skimage.morphology as skimg_morph
import skimage 
import cv2
import pyransac3d as pyrsc

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
            image = batch[1]
            orig_image = batch[2]

            H, W = data.shape[-2:]
            out = self.model(data.unsqueeze(0), None)
            obj_logits = out[0]
            alphas = out[-1]
            ## mask
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

            print(x, y, w, h)

            cam_numpy = cv2.applyColorMap(cam_numpy, cv2.COLORMAP_JET)
            img_numpy = image.permute(1,2,0).numpy()

            img = Drawer.draw_heatmap(cam_numpy, img_numpy)
            
            label, path = image_path.split('/')[-2:]
            image_id = path.split('.')[0] ## str
            img = cam_numpy*0.5 + img_numpy*0.3
            img = np.uint8(img)

            ## draw box
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2, cv2.LINE_AA)

            ## write out
            filename = os.path.join(draw_path, "{}_{}.png".format(image_id, pred.item()))
            cv2.imwrite(filename, img)

            
def make_meshgrid(H0, W0):
    x = np.linspace(-200,200,W0)
    y = np.linspace(-200,200,H0)
    xv, yv = np.meshgrid(x, y)
    grid = np.stack([xv, yv], axis=-1)
    return grid

def fitPlaneToPoints(points):
    # print(points.shape)
    center = np.mean(points, axis=0)
    ## subtract center
    points -= center
    ## covariance
    X = np.matmul(np.transpose(points), points)
    ## PCA
    eigvals, eigvecs = np.linalg.eig(X)
    plane_normal = eigvecs[:, eigvals.argmin()]
    return plane_normal, center

def computePoint2PlaneDistance(points, plane_normal, plane_center):
    v = np.transpose(points - plane_center)
    error = np.fabs(np.matmul(plane_normal, v))
    # print(np.median(error))
    # label_map = error < 0.3
    return error

def computePoint2PlaneDistance2(points, plane_func):
    ## point projection root to plane
    pc_Z = -(np.matmul(points[:, 0:2], plane_func[0:2, :]) + plane_func[-1, :]) / plane_func[2, :]
    plane_xyz = points.copy()
    plane_xyz[:, 2:3] = pc_Z
    ## point to plane dist
    dist = plane_xyz - points
    dist = np.sqrt(np.sum(dist**2, axis=1)+10e-10)
    return dist


def return_grasping_info(points):
    # print(points.shape)
    center = np.mean(points, axis=0)
    ## subtract center
    points -= center
    ## covariance
    X = np.matmul(np.transpose(points), points)
    ## PCA
    eigvals, eigvecs = np.linalg.eig(X)
    direction = eigvecs[:, eigvals.argmax()]
    return center, direction 


def convertPixelsToXYZ(pixels, depth):
    pixels = pixels.reshape(-1, 2)
    depth = depth.reshape(-1)
    depth = depth.astype(np.float) / 1000
    camera_info = [616.7815551757812, 0.0, 328.0075378417969, 0.0, 616.3272705078125, 233.31553649902344, 0.0, 0.0, 1.0]
    x = (pixels[:, 0] - camera_info[2]) / camera_info[0] 
    y = (pixels[:, 1] - camera_info[5]) / camera_info[4]
    x = depth * x
    y = depth * y
    points = np.stack([x, y, depth], axis=-1) 
    return points    


class DepthDetector(object):
    def __init__(self, model, data_loader, device):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.np_grid = make_meshgrid(H0=480, W0=640)
        plane = [-0.02391823687059611, 0.4708401653868865, 0.8818942434348076, -0.3529257165636791]
        self.base_plane = np.array(plane).reshape(-1, 1)
        self.plane_model = pyrsc.Plane()


    def __call__(self, image_path, depth_path, draw_path):
        with torch.no_grad():
            batch = self.data_loader(image_path)
            data = batch[0].to(self.device)
            image = batch[1]
            orig_image = batch[2]

            filename = image_path.split('/')[-1]

            H0, W0 = orig_image.shape[-2:]
            out = self.model.detect(data.unsqueeze(0))
            obj_logits = out[0]
            obj_attns = out[1] ## cam

            ## pred label
            pred = torch.max(obj_logits, dim=-1)[1]
            confidence = self.model.compute_entropy_weight(obj_logits)

            ## cam 
            cam = obj_attns[pred.item()]
            max_pid = torch.argmax(cam)
            h0, w0 = cam.shape[-2:]

            bh, bw = max_pid.floor_divide(w0), max_pid % h0
            cam_box = [float(bw-1.5)/w0*W0, float(bh-1.5)/h0*H0, float(bw+2.5)/w0*W0, float(bh+2.5)/h0*H0]
            if cam_box[0] < 0:
                cam_box[0] = 0
            if cam_box[1] < 0:
                cam_box[1] = 0
            if cam_box[2] > W0:
                cam_box[2] = W0
            if cam_box[3] > H0:
                cam_box[3] = H0
            cam_box = np.array(cam_box).astype(np.int)


            ## depth
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            depth = np.clip(depth, a_min=350, a_max=650)
            pc = convertPixelsToXYZ(self.np_grid, depth) ## [N, 3]
            
            ## base filtering dist
            pc_patch = pc.reshape(H0, W0, 3)[cam_box[1]:cam_box[3], cam_box[0]:cam_box[2], :]
            pc_patch = pc_patch.reshape(-1, 3)
            dist = computePoint2PlaneDistance2(pc_patch.copy(), self.base_plane)
            base_filter_idx = dist > 0.005      ## mm is the unit
            
            ## extract small plane from pc_patch
            grasp_info = None
            pc_patch_filter_base = pc_patch[base_filter_idx, :]
            if len(pc_patch_filter_base) > 100:
                best_eq, best_inliers = self.plane_model.fit(pc_patch_filter_base.copy(), 0.001, maxIteration=100)
                plane_pts = pc_patch_filter_base[best_inliers,:]

                ## return for grasping
                # grasp_info: [center, direction]
                # center is the geometric center of the extracted point set
                # direction is the line to align the two vacuum suction devices
                grasp_info = return_grasping_info(plane_pts)

                dist = computePoint2PlaneDistance2(pc_patch.copy(), np.array(best_eq).reshape(-1,1))
                label_idx = dist < 0.003
                ## draw labelmap
                label_map = np.zeros((H0, W0))
                tmp_label = np.zeros((cam_box[3]-cam_box[1], cam_box[2]-cam_box[0])).reshape(-1)
                tmp_label[label_idx] = pred+1
                tmp_label = tmp_label.reshape(cam_box[3]-cam_box[1], cam_box[2]-cam_box[0])
                label_map[cam_box[1]:cam_box[3], cam_box[0]:cam_box[2]] = tmp_label


            # vcolor = np.transpose(orig_image.reshape(3, -1))
            # print(pc.shape)
            # with open("pc.obj", 'w') as f:
            #     for (v, c) in zip(pc, vcolor):
            #         f.write(f"v {v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n")

            # # # print('pc_patch')
            # # # with open("pc_patch.obj", 'w') as f:
            # # #     for v in pc_patch:
            # # #         f.write(f"v {v[0]} {v[1]} {v[2]}\n")

            # print('plane_pts')
            # with open("plane_pts.obj", 'w') as f:
            #     for v in plane_pts:
            #         f.write(f"v {v[0]} {v[1]} {v[2]}\n")


            if draw_path is not None:
                cam = F.interpolate(cam.unsqueeze(0), size=(H0, W0), mode='bilinear').squeeze(0)

                ## cam
                cam_numpy = cam.detach().cpu().numpy() ## convert to np
                cam_numpy = np.expand_dims(cam_numpy.squeeze(0), axis=-1)
                cam_numpy = np.clip(cam_numpy*255, a_min=0.0, a_max=255.0)
                cam_numpy = cam_numpy.astype(np.uint8) ## convert to cv data
                cam_numpy = cv2.cvtColor(cam_numpy, cv2.COLOR_RGB2BGR)
                cam_numpy = cv2.applyColorMap(cam_numpy, cv2.COLORMAP_JET)

                ## cam + color_img
                img_numpy = orig_image.permute(1,2,0).numpy()
                cam_img = cam_numpy*0.5+ img_numpy*0.3
                cam_img = cam_img.astype(np.uint8) ## convert to cv data

                # ## depth
                depth = (depth.astype(np.float) - depth.min()) / (depth.max() - depth.min())
                depth = np.clip(depth*255, a_min=0, a_max=255)
                depth = depth.astype(np.uint8)
                depth = cv2.cvtColor(depth, cv2.COLOR_RGB2BGR)
                cam_depth = cam_numpy*0.2+ depth*0.7
                cam_depth = cam_depth.astype(np.uint8) ## convert to cv data 

                if len(pc_patch_filter_base) > 100:
                    # label_image = np.repeat(label_map.reshape(H0, W0, 1), 3, axis=-1)
                    # print("LABEL MAP SHAPE", label_image.shape)
                    # print("cam_img SHAPE", cam_img.shape)
                    label_image = skimage.color.label2rgb(label_map, image=depth, bg_label=0)
                    label_image = label_image*255
                    label_image = label_image.astype(np.uint8)
                    cat_img = np.concatenate((cam_img, label_image), axis=1)
                else:
                    cat_img = np.concatenate((cam_img, cam_depth), axis=1)

                cv2.putText(cat_img, f'{pred.item()}-{confidence.item():4.4f}', (200,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) 

                filename = os.path.join(draw_path, filename)
                cv2.imwrite(filename, cat_img)

            
            return grasp_info

        



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
    # detector = NewDetector(model, data_loader, device)
    detector = DepthDetector(model, data_loader, device)

    metafile = 'metadata/rgbd_conveyor_2021-01-16-19-12-28.json'
    files = read_json(metafile)
    image_paths = []
    depth_paths = []

    for idx, f in enumerate(files):
        if f[-5] == 'c' and files[idx+1][-5] == 'd':
            image_paths.append(f)
            depth_paths.append(files[idx+1])
    
    print("# color images", len(image_paths))
    print("# color images", len(depth_paths))
    
    # foldername = '../rgbd_conveyor_2021-01-16-19-12-28'
    
    # rgb_path = '1610795560.31780028_c.png'
    # depth_path = '1610795560.38460255_d.png'

    # # rgb_path = '1610795580.89956570_c.png'
    # # depth_path = '1610795581.03297043_d.png'

    # # rgb_path = '1610795568.35703564_c.png'
    # # # rgb_path = '1610795568.25693083_c.png'
    # # depth_path = '1610795568.49045515_d.png'
    # # # depth_path = '1610795568.32381868_d.png'

    # rgb_path = '1610795584.33534336_c.png'
    # depth_path = '1610795584.46876097_d.png'

    # rgb_path = os.path.join(foldername, rgb_path)
    # depth_path = os.path.join(foldername, depth_path)
    # detector(rgb_path, depth_path, draw_path=save_path)

    for idx, path in tqdm(
        enumerate(image_paths), 
        total=len(image_paths),
        ncols = 80,
        desc= f'detection',
        ):
        print(path)
        detector(path, depth_paths[idx], draw_path=save_path)
        
    

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