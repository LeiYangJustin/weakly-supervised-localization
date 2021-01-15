import json
from collections import OrderedDict
import torch
import torch.nn.functional as F 
import cv2
import numpy as np
import glob
import os
from os.path import isfile, join

## json file IO
def read_json(fname):
    with open(fname, 'rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    with open(fname, 'wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def write_json_no_indent(content, fname):
    with open(fname, 'wt') as handle:
        json.dump(content, handle, sort_keys=False)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SimpleLogger(object):
    def __init__(self, kwarg_list):
        self.kwarg_list = kwarg_list
        self.content = {}
        for kwarg in kwarg_list:
            self.content[kwarg] = []

    def update(self, loss, kwarg):
        assert kwarg in self.kwarg_list
        self.content[kwarg].append(loss)

    def get_best(self, kwarg, best='min'):
        assert kwarg in self.kwarg_list
        if best == 'min':
            best_val = min(self.content[kwarg])
        else:
            best_val = max(self.content[kwarg])

        return self.content[kwarg].index(best_val)


class Drawer:
    @staticmethod
    def normalize_data_to_img255(data):
        min_val = np.amin(data)
        max_val = np.amax(data)
        # RESCALING THE DATA to 0 255
        img = (data - min_val) / (max_val+10e-5-min_val) * 255
        img = img.astype(np.uint8)
        return img

    @staticmethod
    def cv_write_numpy_to_image(img, fname, debug=False):
        fname = fname+".png"
        cv2.imwrite(fname, img)
        if debug:
            print(f"draw {fname}")

    @staticmethod
    def resize_image(img, size):
        img_resized = cv2.resize(img, size)
        return img_resized
    
    @staticmethod
    def write_text_to_img(img, text):
        # Write some Text
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (100, 100)
        fontScale              = 1
        fontColor              = (255, 255, 255)
        lineType               = 2

        cv2.putText(img, text,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        return img


    @staticmethod
    def draw_heatmap(data, src, path=None, normalized=False):
        reshape_size = src.shape
        data = Drawer.resize_image(data, (reshape_size[1],reshape_size[0]))
        heatmap = Drawer.normalize_data_to_img255(data)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        img = 0.5*src + 0.3*heatmap
        if path is None:
            return img
        else:
            Drawer.cv_write_numpy_to_image(img, path)

    
    @staticmethod
    def draw_src_and_reconstructed_image(src, recon, path=None):
        reshape_size = src.shape
        recon = Drawer.resize_image(recon, (reshape_size[1],reshape_size[0]))
        img = np.concatenate([src, recon], axis=0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if path is None:
            return img
        else:
            Drawer.cv_write_numpy_to_image(img, path)



    @staticmethod
    def draw_boxes_on_image(boxes, data, cids, path=None, convert=False):
        """
        data: numpy array
        boxes: list of size [N, [4]]; or numpy array of shape [N, 4]
        """
        print("IN DRAWING ", boxes.shape)
        colors = [(0,0,255), (0,255,0), (255,0,0), (128,128,128)]
        thickness = 2
        ## convert to cv.UMat
        if convert:
            img = cv2.cvtColor(data, cv2.COLOR_RGB2BGR) 
        else:
            img = data.copy()
        for idx, box in zip(cids, boxes):
            tl = box[:2]
            br = box[2:]
            img = cv2.rectangle(img, tuple(tl), tuple(br), colors[idx], thickness)
        
        print("drawing done")
        print(img.shape)
        if path is None:
            return img
        else:
            Drawer.cv_write_numpy_to_image(img, path)

        
    @staticmethod
    def write_video_from_images(pathIn, videoName, deleteImg=True):
        frame_array = []
        files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
        # for sorting the file names properly
        files.sort(key=lambda x: x[5:-4])
        files.sort()

        size = (0, 0)
        img_list = []
        for filename in files:
            if filename.split('.')[-1] == 'png':
                print(filename)
                img_list.append(filename)
                img = cv2.imread(os.path.join(pathIn, filename))
                height, width, layers = img.shape
                size = (width, height)
                text = filename[-16:]
                Drawer.write_text_to_img(img, text)
                frame_array.append(img)
            
        print(size)
        out = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        for i in range(len(frame_array)):
            out.write(frame_array[i])

        out.release()
        print("finishing writing the video")

        if deleteImg:
            for f in img_list:
                os.remove(os.path.join(pathIn, f))

    @staticmethod
    def get_frames_from_a_video(vidfile, save_path):
        """
        :param vidfile: video file
        :param segs: video segment list: [(start, end)] (second)
        :param sample_num: sampling number for each segment. +num: fix sample num. 0: 1 fps. -1: 16 fps
        """
        if os.path.exists(vidfile):
            video = cv2.VideoCapture(vidfile)
        else:
            raise IOError
        
        reshape_size = (455, 256) ## 1920x1080
        
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count/fps

        print('fps = ' + str(fps))
        print('number of frames = ' + str(frame_count))
        print('duration (S) = ' + str(duration))
        minutes = int(duration/60)
        seconds = duration%60
        print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))

        # Grab a few frames
        count = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            # resize
            if reshape_size is not None:
                frame = cv2.resize(frame, reshape_size)
            frame_name = 'new_test'+f'{count:04d}'
            cv2.imwrite(os.path.join(save_path, frame_name+'.png'), frame)
            count += 1
        
        video.release()