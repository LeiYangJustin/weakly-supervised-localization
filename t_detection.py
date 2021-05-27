import time as time
from tqdm import tqdm

import torch 
import torch.nn as nn
from torchvision import transforms
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

from helper import plt_show
from mynet import StreamingDataloader 
from mynet import MyNet, EDModel, BoxOptimizer
from utils import Drawer, read_json
import box_finder
import time


BOX_SCORE_THRESHOLD = 0.2

package_dir = os.path.abspath(os.path.dirname(__file__))
resume_default = os.path.join(package_dir, './ckpt/checkpoint-epoch99.pth')
# resume_default = os.path.join(package_dir, './2021-05-27_08-16-24/model/checkpoint-epoch80.pth')

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
    print(f"cams.shape: {cams.shape}")
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


def make_meshgrid(H0, W0):
    x = np.linspace(0, W0, W0)
    y = np.linspace(0, H0, H0)
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
    normal = eigvecs[:, eigvals.argmin()]
    # mat44=np.eye(4)
    # mat44[:3,:3] = eigvecs
    # mat44[:3, 3] = center
    # return mat44
    return center, direction, normal, len(points)


def convertPixelsToXYZ(pixels, depth):
    pixels = pixels.reshape(-1, 2)
    depth = depth.reshape(-1)
    depth = depth.astype(np.float) / 1000
    camK = [616.7815551757812, 0.0, 328.0075378417969, 0.0, 616.3272705078125, 233.31553649902344, 0.0, 0.0, 1.0]
    x = (pixels[:, 0] - camK[2]) / camK[0]
    y = (pixels[:, 1] - camK[5]) / camK[4]
    x = depth * x
    y = depth * y
    points = np.stack([x, y, depth], axis=-1) 
    return points    


class DepthDetector(object):
    def __init__(self, model, data_loader, device, plane):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.np_grid = make_meshgrid(H0=480, W0=640)
        # self.np_grid = make_meshgrid(H0=640, W0=480)
        # plane = [-0.02391823687059611, 0.4708401653868865, 0.8818942434348076, -0.3529257165636791]
        # plane = [-0.01983893474403382, 0.4804387040072502, 0.8768038939010568, -0.4349528512188574]
        # plane = [-0.004408289548550268, -0.48735298696910767, -0.8731939263849381, 0.443669926687439]
        # plane = [0.0012687906468578781, -0.5038659049335062, -0.8637809560391067, 0.4396400570752005]
        # plane = [0.002052785611812755, -0.49563451617902893, -0.8685287631640141, 0.44178835957612994]
        if plane is None:
            plane = [-0.0031245003473275735, -0.48572054764360334, -0.8741085671095884, 0.41095290596847167]
        self.base_plane = np.array(plane).reshape(-1, 1)
        self.plane_model = pyrsc.Plane()
        self.imwidth = 224

        self.normalize = transforms.Normalize(mean=[0.5084, 0.4224, 0.3769],
                                              std=[0.2599, 0.2371, 0.2323])
        self.initial_transforms = transforms.Compose([transforms.Resize((self.imwidth, self.imwidth))])

    def run_with_cvimg_input_with_obj_id(self, obj_id, rgb_image, depth_image, draw_path=None):

        with torch.no_grad():

            data = torch.from_numpy(rgb_image).permute(2,0,1) ## [H, W, C] --> [C, H, W]
            data=data.to(torch.float32).to(device=self.device)

            ## convert RGB image to 0-1 scale and normalize it using the ImageNet mean and std
            data = data/255.0
            data = self.normalize(data)
            data = self.initial_transforms(data) ## keep this line; use torchvision latest version

            H0, W0 = rgb_image.shape[:2]
            out = self.model.detect(data.unsqueeze(0))
            obj_logits = out[0]
            obj_attns = out[1]  ## cam

            ## pred label
            pred = obj_id
            # confidence = self.model.compute_entropy_weight(obj_logits)
            confidence = F.softmax(obj_logits/10)
            predicted_id = torch.argmax(obj_logits.squeeze())
            predicted_confidence = torch.max(confidence.squeeze())
            target_confidence = confidence.squeeze()[obj_id]

            print("obj_logits", obj_logits)
            print("target", pred)
            print("predicted_id", predicted_id.item())
            print("confidence", target_confidence, predicted_confidence)
            
            ## cam
            cam = obj_attns[pred]
            max_pid = torch.argmax(cam)
            h0, w0 = cam.shape[-2:]

            bh, bw = max_pid.floor_divide(w0), max_pid % h0
            print("cam center", bh, bw)
            cam_box = [float(bw - 1.0) / w0 * W0, float(bh - 1.0) / h0 * H0, float(bw + 2.0) / w0 * W0,
                       float(bh + 2.0) / h0 * H0]
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
            depth = depth_image
            # depth = np.clip(depth, a_min=350, a_max=650) ## seem no need

            pc = convertPixelsToXYZ(self.np_grid, depth)  ## [N, 3]
            # best_eq, best_inliers = self.plane_model.fit(pc.copy(), 0.001, maxIteration=100)
            # print(best_eq)
            # estimate_base_plane_mode = True
            # estimate_base_plane_mode = False
            # if estimate_base_plane_mode:
            #     pc_patch = pc.reshape(H0, W0, 3)#[cam_box[1]:cam_box[3], cam_box[0]:cam_box[2], :]
            #     pc_patch = pc_patch.reshape(-1, 3)
            #     best_eq, best_inliers = self.plane_model.fit(pc_patch.copy(), 0.0005, maxIteration=100)
            #
            #     print("base", self.base_plane)
            #     best_eq = np.array(best_eq).reshape(-1, 1) ## [4, 1]
            #     if best_eq[-1, 0] < 0:
            #         best_eq = best_eq*-1.0
            #     print("estimated", list(best_eq.reshape(-1)))
            #     cur_base_plane = 0.9*self.base_plane + 0.1*best_eq
            #     dist = computePoint2PlaneDistance2(pc_patch.copy(), self.base_plane)
            #     print(np.mean(dist), np.max(dist), np.min(dist), np.median(dist))
            #     dist = computePoint2PlaneDistance2(pc_patch.copy(), best_eq)
            #     print(np.mean(dist), np.max(dist), np.min(dist), np.median(dist))
            #     dist = computePoint2PlaneDistance2(pc_patch.copy(), cur_base_plane)
            #     print(np.mean(dist), np.max(dist), np.min(dist), np.median(dist))
            #
            #     self.base_plane = cur_base_plane.copy()
            #
            #     return (None, None, None, None)

            ## base filtering dist
            pc_patch = pc.reshape(H0, W0, 3)[cam_box[1]:cam_box[3], cam_box[0]:cam_box[2], :]
            pc_patch = pc_patch.reshape(-1, 3)

            dist = computePoint2PlaneDistance2(pc_patch.copy(), self.base_plane)
            base_filter_idx = (dist > 0.01)*(dist < 0.08)   ## m is the unit

            ## extract small plane from pc_patch
            conf_threshold = 0.95
            grasp_info = None
            pc_patch_filter_base = pc_patch[base_filter_idx, :]
            
            # remove invalid depth
            valid_idx = np.argwhere(pc_patch_filter_base[:,2] != 0).reshape(-1)
            pc_patch_filter_base=pc_patch_filter_base[valid_idx]
            
            if len(pc_patch_filter_base) > 1000:
                best_eq, best_inliers = self.plane_model.fit(pc_patch_filter_base.copy(), 0.0001, maxIteration=100)

                plane_pts = pc_patch_filter_base[best_inliers, :]
                
                # print("number of plane pts: ", len(plane_pts))
                # dist = computePoint2PlaneDistance2(plane_pts.copy(), self.base_plane)
                # print(np.mean(dist), np.max(dist), np.min(dist), np.median(dist))

                ## return for grasping
                # grasp_info: [center, direction, normal, len(points)]
                # center is the geometric center of the extracted point set
                # direction is the line to align the two vacuum suction devices
                grasp_info = return_grasping_info(plane_pts)

                dist = computePoint2PlaneDistance2(pc_patch.copy(), np.array(best_eq).reshape(-1, 1))
                label_idx = dist < 0.003

                ##
                label_pts = pc_patch[label_idx, :]
                print("number of plane pts: ", len(label_pts))
                dist = computePoint2PlaneDistance2(label_pts.copy(), self.base_plane)
                print(np.mean(dist), np.max(dist), np.min(dist), np.median(dist))

                ## draw labelmap
                label_map = np.zeros((H0, W0))
                tmp_label = np.zeros((cam_box[3] - cam_box[1], cam_box[2] - cam_box[0])).reshape(-1)
                tmp_label[label_idx] = pred + 1
                tmp_label = tmp_label.reshape(cam_box[3] - cam_box[1], cam_box[2] - cam_box[0])
                label_map[cam_box[1]:cam_box[3], cam_box[0]:cam_box[2]] = tmp_label

            # ------------------------

            """ plot all three cams for debugging
            for idx, attn_map in enumerate(obj_attns):
                tmp_cam = F.interpolate(attn_map.unsqueeze(0), size=(H0, W0), mode='bilinear').squeeze(0)
                cam_numpy = tmp_cam.detach().cpu().numpy()  ## convert to np
                cam_numpy = np.expand_dims(cam_numpy.squeeze(0), axis=-1)
                cam_numpy = np.clip(cam_numpy * 255, a_min=0.0, a_max=255.0)
                cam_numpy = cam_numpy.astype(np.uint8)  ## convert to cv data
                cam_numpy = cv2.cvtColor(cam_numpy, cv2.COLOR_RGB2BGR)
                cam_numpy = cv2.applyColorMap(cam_numpy, cv2.COLORMAP_JET)
                ## cam + color_img
                cam_numpy = cam_numpy * 0.5 + rgb_image * 0.3
                cam_numpy = cam_numpy.astype(np.uint8)  ## convert to cv data
                print(idx)
                if idx == 0:
                    cam_img = cam_numpy.copy()
                    print(cam_img.shape)
                else:
                    cam_img = np.concatenate((cam_img, cam_numpy), axis=1)
                    print(cam_img.shape)
        
            filename = 'tmp_cam.png'
            cv2.imwrite(filename, cam_img)
            input()
            """


            """ COMMENT OUT FOR NOW """
            cam = F.interpolate(cam.unsqueeze(0), size=(H0, W0), mode='bilinear').squeeze(0)

            ## cam
            cam_numpy = cam.detach().cpu().numpy()  ## convert to np
            cam_numpy = np.expand_dims(cam_numpy.squeeze(0), axis=-1)

            if target_confidence.item() > conf_threshold:
                cam_numpy = np.clip(cam_numpy * 255, a_min=0.0, a_max=255.0)
            else:
                cam_numpy = np.clip(cam_numpy * 255, a_min=0.0, a_max=0.0)
            cam_numpy = cam_numpy.astype(np.uint8)  ## convert to cv data
            cam_numpy = cv2.cvtColor(cam_numpy, cv2.COLOR_RGB2BGR)
            cam_numpy = cv2.applyColorMap(cam_numpy, cv2.COLORMAP_JET)

            ## cam + color_img
            cam_img = cam_numpy * 0.5 + rgb_image * 0.3
            cam_img = cam_img.astype(np.uint8)  ## convert to cv data

            ## depth
            # depth = np.clip(depth, a_min=350, a_max=650) ## seem no need
            depth = (depth.astype(np.float) - depth.min()) / (depth.max() - depth.min())
            depth = np.clip(depth * 255, a_min=0, a_max=255)
            depth = depth.astype(np.uint8)
            depth = cv2.cvtColor(depth, cv2.COLOR_RGB2BGR)
            cam_depth = cam_numpy * 0.2 + depth * 0.7
            cam_depth = cam_depth.astype(np.uint8)  ## convert to cv data

            if grasp_info is not None:
                # label_image = np.repeat(label_map.reshape(H0, W0, 1), 3, axis=-1)
                # print("LABEL MAP SHAPE", label_image.shape)
                # print("cam_img SHAPE", cam_img.shape)
                label_image = skimage.color.label2rgb(label_map, image=depth, bg_label=0)
                label_image = label_image * 255
                label_image = label_image.astype(np.uint8)
                cat_img = np.concatenate((cam_img, label_image), axis=1)
            else:
                cat_img = np.concatenate((cam_img, cam_depth), axis=1)
            # cat_img = cam_img
            cv2.putText(cat_img, f'target id: {obj_id}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

            if predicted_confidence.item() > conf_threshold:
                cv2.putText(cat_img, f'predicted id: {predicted_id}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(cat_img, f'predicted id: -1', (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.putText(cat_img, f'{pred}-{confidence.item():4.4f}', (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #             (255, 255, 255), 2, cv2.LINE_AA)
            
            if draw_path is not None:
                filename = 'tmp.png'
                cv2.imwrite(filename, cat_img)

            return obj_id, target_confidence.item(), grasp_info, cat_img

    #
    # def run_with_cvimg_input(self, rgb_image, depth_image, draw_path=None):
    #     with torch.no_grad():
    #
    #         data = torch.from_numpy(rgb_image).permute(2,0,1)
    #         data=data.to(torch.float32).to(device=self.device)
    #
    #         ## convert RGB image to 0-1 scale and normalize it using the ImageNet mean and std
    #         data = data/255.0
    #         data = self.normalize(data)
    #         data = self.initial_transforms(data)
    #
    #
    #         H0, W0 = rgb_image.shape[:2]
    #         out = self.model.detect(data.unsqueeze(0))
    #         obj_logits = out[0]
    #         obj_attns = out[1]  ## cam
    #
    #         ## pred label
    #         pred = torch.max(obj_logits, dim=-1)[1].cpu()
    #         confidence = self.model.compute_entropy_weight(obj_logits)
    #         print("obj_logits", obj_logits)
    #         print("pred", pred)
    #         print("confidence", confidence)
    #
    #         ## cam
    #         cam = obj_attns[pred.item()]
    #         max_pid = torch.argmax(cam)
    #         h0, w0 = cam.shape[-2:]
    #
    #         bh, bw = max_pid.floor_divide(w0), max_pid % h0
    #         cam_box = [float(bw - 1.5) / w0 * W0, float(bh - 1.5) / h0 * H0, float(bw + 2.5) / w0 * W0,
    #                    float(bh + 2.5) / h0 * H0]
    #         if cam_box[0] < 0:
    #             cam_box[0] = 0
    #         if cam_box[1] < 0:
    #             cam_box[1] = 0
    #         if cam_box[2] > W0:
    #             cam_box[2] = W0
    #         if cam_box[3] > H0:
    #             cam_box[3] = H0
    #         cam_box = np.array(cam_box).astype(np.int)
    #
    #         ## depth
    #         depth = depth_image
    #         depth = np.clip(depth, a_min=350, a_max=650)
    #         pc = convertPixelsToXYZ(self.np_grid, depth)  ## [N, 3]
    #         # best_eq, best_inliers = self.plane_model.fit(pc.copy(), 0.001, maxIteration=100)
    #         # print(best_eq)
    #
    #         ## base filtering dist
    #         pc_patch = pc.reshape(H0, W0, 3)[cam_box[1]:cam_box[3], cam_box[0]:cam_box[2], :]
    #         pc_patch = pc_patch.reshape(-1, 3)
    #
    #         dist = computePoint2PlaneDistance2(pc_patch.copy(), self.base_plane)
    #         base_filter_idx = dist > 0.005  ## mm is the unit
    #
    #         ## extract small plane from pc_patch
    #         grasp_info = None
    #         pc_patch_filter_base = pc_patch[base_filter_idx, :]
    #         if len(pc_patch_filter_base) > 1000 and confidence > 0.9:
    #             best_eq, best_inliers = self.plane_model.fit(pc_patch_filter_base.copy(), 0.001, maxIteration=100)
    #             plane_pts = pc_patch_filter_base[best_inliers, :]
    #
    #             ## return for grasping
    #             # grasp_info: [center, direction]
    #             # center is the geometric center of the extracted point set
    #             # direction is the line to align the two vacuum suction devices
    #             grasp_info = return_grasping_info(plane_pts)
    #
    #             dist = computePoint2PlaneDistance2(pc_patch.copy(), np.array(best_eq).reshape(-1, 1))
    #             label_idx = dist < 0.003
    #             ## draw labelmap
    #             label_map = np.zeros((H0, W0))
    #             tmp_label = np.zeros((cam_box[3] - cam_box[1], cam_box[2] - cam_box[0])).reshape(-1)
    #             tmp_label[label_idx] = pred + 1
    #             tmp_label = tmp_label.reshape(cam_box[3] - cam_box[1], cam_box[2] - cam_box[0])
    #             label_map[cam_box[1]:cam_box[3], cam_box[0]:cam_box[2]] = tmp_label
    #
    #         # ------------------------
    #         cam = F.interpolate(cam.unsqueeze(0), size=(H0, W0), mode='bilinear').squeeze(0)
    #
    #         ## cam
    #         cam_numpy = cam.detach().cpu().numpy()  ## convert to np
    #         cam_numpy = np.expand_dims(cam_numpy.squeeze(0), axis=-1)
    #         cam_numpy = np.clip(cam_numpy * 255, a_min=0.0, a_max=255.0)
    #         cam_numpy = cam_numpy.astype(np.uint8)  ## convert to cv data
    #         cam_numpy = cv2.cvtColor(cam_numpy, cv2.COLOR_RGB2BGR)
    #         cam_numpy = cv2.applyColorMap(cam_numpy, cv2.COLORMAP_JET)
    #
    #         ## cam + color_img
    #         cam_img = cam_numpy * 0.5 + rgb_image * 0.3
    #         cam_img = cam_img.astype(np.uint8)  ## convert to cv data
    #
    #         # ## depth
    #         depth = (depth.astype(np.float) - depth.min()) / (depth.max() - depth.min())
    #         depth = np.clip(depth * 255, a_min=0, a_max=255)
    #         depth = depth.astype(np.uint8)
    #         depth = cv2.cvtColor(depth, cv2.COLOR_RGB2BGR)
    #         cam_depth = cam_numpy * 0.2 + depth * 0.7
    #         cam_depth = cam_depth.astype(np.uint8)  ## convert to cv data
    #
    #         if grasp_info is not None:
    #             # label_image = np.repeat(label_map.reshape(H0, W0, 1), 3, axis=-1)
    #             # print("LABEL MAP SHAPE", label_image.shape)
    #             # print("cam_img SHAPE", cam_img.shape)
    #             label_image = skimage.color.label2rgb(label_map, image=depth, bg_label=0)
    #             label_image = label_image * 255
    #             label_image = label_image.astype(np.uint8)
    #             cat_img = np.concatenate((cam_img, label_image), axis=1)
    #         else:
    #             cat_img = np.concatenate((cam_img, cam_depth), axis=1)
    #
    #         cv2.putText(cat_img, f'{pred.item()}-{confidence.item():4.4f}', (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #                     (255, 255, 255), 2, cv2.LINE_AA)
    #
    #         if draw_path is not None:
    #             # filename = os.path.join(draw_path, filename)
    #             # cv2.imwrite(filename, cat_img)
    #             plt_show(cat_img)
    #         return pred.item(), confidence.item(), grasp_info, cat_img

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


            # ------------------------
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

            if draw_path is not None:
                filename = os.path.join(draw_path, filename)
                cv2.imwrite(filename, cat_img)
            else:
                return grasp_info, cat_img

def get_detector(resume=resume_default, device="cuda:0",plane=None):
    ## dataloader
    data_loader = StreamingDataloader(imwidth=224)

    ## CNN model
    output_dim = 3
    # model = MyNet(output_dim)
    model = EDModel(output_dim, resnet_type='resnet18')
    ## resume a ckpt
    checkpoint = torch.load(resume, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model = model.to(device)
    print(model)

    ## init a detector
    detector = DepthDetector(model, data_loader, device, plane)
    return detector

## main
if __name__ == '__main__':
    detector=get_detector()
    # foldername = '../rgbd_conveyor_2021-01-16-19-12-28'
    data_folder = './test_data'
    test_data = os.path.join(data_folder, '1610867764.77414274_')
    rgb_path = test_data + 'c.png'
    depth_path = test_data + 'd.png'
    img = cv2.imread(rgb_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

    id_seq = [1,0,2] ## H -> C -> P
    # label, confidence, grasp_info, cat_img = detector.run_with_cvimg_input_with_obj_id(rgb_img, depth_img, draw_path='')
    label, grasp_info, cat_img = detector.run_with_cvimg_input(rgb_img, depth_img, draw_path='')

    # tic = time.time()
    # print(tic)
    # for _ in range(0, 10):
    #     detector.run_with_cvimg_input(rgb_img, depth_img)
    # toc = time.time()
    # print("time consumption", (toc - tic) / 10.)
