
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.ops as tvops
from torch.utils.data import DataLoader
import shutil 
import numpy as np
import cv2
import skimage
import os


class RectangleFitter(nn.Module):
    def __init__(self, B):
        super(RectangleFitter, self).__init__()
        self.B = B
        self.reset_XYWH()
    
    def reset_XYWH(self, B=None):
        if B is None:
            B = self.B
        self.xywh = torch.rand((B, 4), requires_grad=True)

    def set_XYWH(self, XYWH):
        self.xywh = XYWH.requires_grad_(True)

    def get_XYWH(self):
        return self.xywh.clone()


    def forward(self, Q, padding):
        xywh = self.xywh
        qxl = (Q[:, :, 0] - xywh[:, None, 0])
        qyl = (Q[:, :, 1] - xywh[:, None, 1])
        qxu = (xywh[:, None, 0] + torch.abs(xywh[:, None, 2])) - Q[:, :, 0]
        qyu = (xywh[:, None, 1] + torch.abs(xywh[:, None, 3])) - Q[:, :, 1]

        qxl = qxl.pow(2)
        qyl = qyl.pow(2)
        qxu = qxu.pow(2)
        qyu = qyu.pow(2)
        dist = torch.stack([qxl, qyl, qxu, qyu], dim=-1)
        dist = dist.min(dim=-1)[0]

        loss = (padding*dist).sum() / self.B

        """
        counting the inliers
        """
        # qxl = F.tanh(100*qxl)
        # qyl = F.tanh(100*qyl)
        # qxu = F.tanh(100*qxu)
        # qyu = F.tanh(100*qyu)
        # area = torch.abs(xywh[2]) * torch.abs(xywh[3])
        # loss = (qxl*qxu).sum() + (qyl*qyu).sum() - 100*area
        # loss = loss.neg()

        return loss

    def update(self, lr):
        with torch.no_grad():
            self.xywh -= lr * self.xywh.grad
            # Manually zero the gradients after updating weights
            self.xywh.grad = None


def largest_component(image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    return img2.astype('uint8')

"""
all are numpy ndarray
"""
def generate_bbox_and_lcc(image, bar):
    bar = int(image.max()*bar)
    ret,thresh1 = cv2.threshold(image,bar,255,cv2.THRESH_BINARY)
    lcc_img = largest_component(thresh1)
    contours, _ = cv2.findContours(lcc_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) ## contours [N, 1, 2]
    assert len(contours) == 1
    contour = contours[0][:, 0, :2]

    box_min = np.amin(contour, axis=0)
    box_max = np.amax(contour, axis=0)
    box = np.concatenate((box_min, box_max), axis=0)
    return box, contour



class BoxOptimizer():
    """
    bmask is a tensor
    """
    def __init__(self, batch_size, max_iter=100, contour_thres=0.5, learning_rate=0.001):
        self.max_iter = max_iter
        self.lr = learning_rate
        self.contour_thres = contour_thres
        self.model = RectangleFitter(batch_size)
        

    def __call__(self, batch_mask):
        B, H, W = batch_mask.shape

        ## get boxes and lcc_contours from masks
        batch_box = []
        lcc_contour_list = []
        max_length = 0
        for bmask in batch_mask:
            box, lcc_contour = generate_bbox_and_lcc(np.array(bmask), bar=self.contour_thres)
            batch_box.append(torch.from_numpy(box))
            lcc_contour_list.append(torch.from_numpy(lcc_contour))
            max_length = len(lcc_contour)
        
        batch_box = torch.stack(batch_box).to(torch.float32) ## [B, 4]
        batch_lcc_contour = torch.zeros((B, max_length, 2))
        batch_padding = torch.zeros((B, max_length))
        for b, lcc_c in enumerate(lcc_contour_list):
            length = len(lcc_c)
            batch_lcc_contour[b, :length] = lcc_c
            batch_padding[b, :length] = 1

        ##
        batch_box[:,0] = batch_box[:,0]/W
        batch_box[:,1] = batch_box[:,1]/H
        batch_box[:,2] = batch_box[:,2]/W
        batch_box[:,3] = batch_box[:,3]/H
        batch_box[:,2] = batch_box[:,2] - batch_box[:,0]
        batch_box[:,3] = batch_box[:,3] - batch_box[:,1]


        ## [B, N, 2]
        batch_lcc_contour[:, :, 0] = batch_lcc_contour[:, :, 0]/H
        batch_lcc_contour[:, :, 1] = batch_lcc_contour[:, :, 1]/W
        
        ## need a good initialization
        self.model.set_XYWH(batch_box)
        for idx in range(self.max_iter):
            ## forward pass
            loss = self.model(batch_lcc_contour, batch_padding)

            ## run backward pass
            loss.backward()
            self.model.update(self.lr)

        return self.model.get_XYWH(), lcc_contour_list




if __name__ == "__main__":


    if not os.path.exists("res"):
        os.mkdir("res")

    img_rgb = cv2.imread("cam.png")
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    H, W = img.shape

    batch_size = 6
    batch_mask = torch.from_numpy(img)
    batch_mask = batch_mask.expand(batch_size, -1, -1)

    model = BoxOptimizer(batch_size)
    xywh, lcc_contour_list = model(batch_mask)

    batch_mask_detach = batch_mask.detach()

    for b in range(batch_size):

        src = np.uint8(img_rgb.copy())
        x, y, w, h = xywh.detach()[b]

        print(x, y, w, h)

        x, w = int(x*W), int(np.fabs(w)*W)
        y, h = int(y*H), int(np.fabs(h)*H)

        contour = lcc_contour_list[b]
        N = contour.shape[0]
        contour = contour.reshape(N, 1, 2)
        contour = contour.numpy()
        src=cv2.drawContours(src, contour, -1, (255,255,0), 2)
        cv2.rectangle(src, (x, y), (x+w, y+h), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(f"res/res_{b}.png", src)

    


        