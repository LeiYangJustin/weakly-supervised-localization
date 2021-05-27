import sys
import time

import cv2
import cv2 as cv
import numpy as np

from helper import plt_show


class CircleDetector():
    def __init__(self):
        ## use HSV color template for detecting the specified colors
        self.red_template = [np.array([0, 110, 110]), np.array([30, 255, 255])]
        self.blue_template = [np.array([100, 100, 100]), np.array([140, 255, 255])]
        self.kernel5 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        params = cv2.SimpleBlobDetector_Params()
        # params.minThreshold = 100  # 50
        # params.maxThreshold = 150  # 220
        # Set Area filtering parameters
        params.filterByArea = True
        params.minArea = 3.14 * 3 ** 2
        # Set Circularity filtering parameters
        params.filterByCircularity = True
        params.minCircularity = 0.4
        # # Set Convexity filtering parameters
        # params.filterByConvexity = True
        # params.minConvexity = 0.2
        # Set inertia filtering parameters
        params.filterByInertia = True
        params.minInertiaRatio = 0.5
        # Create a detector with the parameters
        self.detector = cv.SimpleBlobDetector_create(params)

    def __call__(self, img_bgr):
        gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
        # plt_show(gray)
        time_start = time.time()
        ## CONVERT TO HSV FROM BGR
        # hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
        # plt_show(img_bgr[:,:,0])
        # # plt_show(img_bgr[:,:,1])
        # plt_show(img_bgr[:,:,2])
        # gray = img_bgr[:,:,0]
        #

        # Detect blobs
        keypoints = self.detector.detect(gray)
        # keypoints[0].size  zhijing
        print(f"==========================Detect_t:{len(keypoints)}, {time.time()-time_start}")

        # Draw blobs on our image as red circles
        blank = np.zeros((1, 1))
        # blobs = cv2.drawKeypoints(white_bg, keypoints, blank, (0, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        blobs = img_bgr.copy()
        if len(keypoints) < 1:
            print("no circle blob detected")
        for keypoint in keypoints:
            blobs = cv2.circle(blobs, (int(keypoint.pt[0]), int(keypoint.pt[1])), int(keypoint.size/2.0), color=(0,0,0), thickness=-1)

        # # plt_show(blobs)
        # cv.imshow("gray", blobs)
        # cv.waitKey(3)
        # return keypoints
        return blobs, 'none'

def get_circle_detector():
    return CircleDetector()

def main(argv):
    # default_file = 'circle_samples/2019-Dated-U.S.-Coins.jpg'
    # default_file = 'circle_samples/b078wt6js9.main.jpg'
    default_file = 'test_data/210517_141405.962206_c.png'
    # default_file = 'test_data/210517_145405.229187_c.png'
    # default_file = 'test_data/210517_150627.417060_c.png'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(filename, cv.IMREAD_UNCHANGED)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1

    circle_detector = CircleDetector()
    keypoints = circle_detector(src)

    return 0

if __name__ == "__main__":
    main(sys.argv[1:])