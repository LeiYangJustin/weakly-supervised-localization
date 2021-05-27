# import selectivesearch
import numpy as np
import skimage.data
import cv2
import torch
import torch.nn.functional as F



def EdgeBox(image, normalized=False, box_num=50, model_path="model.yml.gz"):
    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model_path)
    rgb_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)
    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)
    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(box_num)
    boxes, objectness = edge_boxes.getBoundingBoxes(edges, orimap) ## [x,y,w,h], [objectness]

    width, height = rgb_im.shape[0], rgb_im.shape[1]
    boxes = np.array(boxes).astype(float)

    boxes[:,2:] = boxes[:,:2]+boxes[:,2:]
    # good_boxes = [b for b in boxes if b[2]/b[3] > 3 or b[3]/b[2] > 3]
    objectness = np.array(objectness)
    if normalized:
        boxes[:,0] = boxes[:,0]/width
        boxes[:,2] = boxes[:,2]/width
        boxes[:,1] = boxes[:,1]/height
        boxes[:,3] = boxes[:,3]/height

    return boxes, objectness

def customized_selective_search(img, normalized=False):
    assert img.shape[-1] == 3, "imgs.shape should be [B, H, W, 3]"
    ss_boxes = []
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=10)
    
    width, height = img.shape[0], img.shape[1]

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 500 pixels
        if r['size'] < 200:
            continue
        x, y, w, h = r['rect']
        if w < 5 or h < 5 or w / h > 2.0 or h / w > 2.0:
            continue
        candidates.add(r['rect'])
        
    data = [list(a) for a in candidates]
    data = np.array(data)
    if data.shape[0] == 0:
        print("this image has no proposals")
    else:
        data[:,2:] = data[:,:2]+data[:,2:]
        data = data.astype(float)
        if normalized:
            data[:,0] = data[:,0]/width
            data[:,2] = data[:,2]/width
            data[:,1] = data[:,1]/height
            data[:,3] = data[:,3]/height

    return data


"""
img_numpy: numpy array [H, W, C]
return:
float type boxes in tensor
"""
def find_boxes(img_numpy, target_image_width, mode='edge_box', box_num=50):
    ## find boxes
    width, height = img_numpy.shape[0:2]
    ratio = float(height)/width
    
    if target_image_width is not None:
        image_size_for_ss_box = (target_image_width, int(target_image_width*ratio))
        src = img_numpy
        img_numpy = cv2.resize(src, image_size_for_ss_box)

    if mode == 'edge_box':
        boxes_float, _ = EdgeBox(img_numpy, normalized=True, box_num=box_num)
    elif mode == 'ss_box':
        boxes_float = customized_selective_search(img_numpy, normalized=True)
    else:
        raise NotImplementedError
    return torch.from_numpy(boxes_float).to(torch.float32)

