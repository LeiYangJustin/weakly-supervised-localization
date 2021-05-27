import sys

import cv2
import cv2 as cv
import numpy as np
from helper import plt_show

from skimage.filters import sobel
from skimage.measure import label
from skimage.segmentation import watershed
from skimage.color import label2rgb
from matplotlib import pyplot as plt

class BlobCircleDetector():
    def __init__(self):
        ## use HSV color template for detecting the specified colors
        # self.red_template = [np.array([0, 110, 110]), np.array([30, 255, 255])]
        # self.blue_template = [np.array([100, 100, 100]), np.array([140, 255, 255])]
        self.red_template = np.array([0, 0, 255])
        self.blue_template = np.array([255, 0, 0])
        self.kernel5 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

        params = cv2.SimpleBlobDetector_Params()
        # params.minThreshold = 100  # 50
        # params.maxThreshold = 150  # 220
        # Set Area filtering parameters
        params.filterByArea = True
        params.minArea = 3.14 * 5 ** 2
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
        # time_start = time.time()

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
        # print(f"==========================Detect_t:{len(keypoints)}, {time.time()-time_start}")

        # Draw blobs on our image as red circles
        blank = np.zeros((1, 1))
        H, W = img_bgr.shape[0:2]
        print(f"{len(keypoints)} circle blob detected")

        if len(keypoints) < 1:
            return img_bgr, 'none'

        max_area = 0
        for idx, keypoint in enumerate(keypoints):
            mask = np.zeros((H, W))
            mask = cv2.circle(
                mask,
                (int(keypoint.pt[0]), int(keypoint.pt[1])),
                int(keypoint.size/2.0),
                color=(1,1,1), thickness=-1)
            blobs = np.expand_dims(mask, axis=-1) * img_bgr
            color_loc = np.argwhere(mask)
            area =  len(color_loc)
            if area > max_area:
                max_area = area
                color_2 = np.array([0,0,0])
                for loc in color_loc:
                    color_2 = color_2 + blobs[loc[0],loc[1], :]
                color_2 = color_2 / len(color_loc)
                color = np.sum(blobs, axis=(0,1)) / np.sum(mask>0)
                dist_to_red = np.linalg.norm(color - self.red_template)
                dist_to_blue = np.linalg.norm(color - self.blue_template)
                print(dist_to_red, dist_to_blue)
                if dist_to_red < dist_to_blue:
                    color_type = 'red'
                else:
                    color_type = 'blue'
                hsv = cv.cvtColor(color[None, None, :].astype(np.uint8), cv.COLOR_BGR2HSV)
                print(hsv)
                blobs = blobs.astype(np.uint8)

        return blobs, color_type



class CircleDetector():
    def __init__(self):
        ## use HSV color template for detecting the specified colors
        self.red_template = [np.array([0, 110, 110]), np.array([30, 255, 255])]
        self.blue_template = [np.array([90, 50, 50]), np.array([140, 255, 255])]
        self.kernel5 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    def __call__(self, img_bgr):

        """thresholding the grayscale image is not robust enough
        ## SMOOTHING THE INPUT
        gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray, 5)
        ret, thresh = cv.threshold(gray, 125, 255, 0) ## 100 to ensure red and green are separatable
        return np.stack([thresh, thresh, thresh], axis=-1)
        """

        mask_dict = {
            "red": [],
            "blue": []
        }

        contour_list = []
        for k, v in mask_dict.items():
            ## CONVERT TO HSV FROM BGR
            hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
            # hsv = cv.medianBlur(hsv, 5)
            if k == 'red':
                color_template = self.red_template
            elif k == "blue":
                color_template = self.blue_template
            else:
                raise NotImplementedError

            ## FIND MASK FOR THE SPECIFIED COLOR (IF ANY)
            mask = cv.inRange(hsv, color_template[0], color_template[1]) ## CV_8U
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, self.kernel5) ## make smooth connected components
            edges = sobel(mask)
            foreground, background = 1, 2
            markers = np.zeros_like(mask)
            markers[mask < 30.0] = background
            markers[mask > 150.0] = foreground
            ws = watershed(edges, markers)

            seg1 = label(ws == foreground).astype(np.uint8)
            white_bg = np.ones_like(img_bgr)*255
            color1 = label2rgb(seg1, image=white_bg, bg_label=0)
            color1 = (color1*255).astype(np.uint8)
            # cv.imshow("gray", color1)
            # cv.waitKey(3)
            # # # input()

            ## COLLECT MASK CONTOURS
            num_regions = np.max(seg1)
            print(np.min(seg1))
            print(np.max(seg1))

            # contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            # for contour in contours:
            for i in range(1,num_regions+1):
                tmp_mask = (seg1==i).astype(np.uint8)
                area = np.sum(tmp_mask)
                # plt_show(tmp_mask)
                contours, _ = cv.findContours(tmp_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if contour.size < 10:
                        print("contour size < 10")
                        continue
                    rot_rect = cv.fitEllipse(contour)
                    cx = int(rot_rect[0][0])
                    cy = int(rot_rect[0][1])
                    sx = int(rot_rect[1][0] / 2)
                    sy = int(rot_rect[1][1] / 2)
                    ellipse_pts = cv.ellipse2Poly((cx, cy), (sx, sy), int(rot_rect[2]), 0, 360, 5)
                    h,w=img_bgr.shape[0],img_bgr.shape[1]
                    black_bg = np.zeros((h,w))
                    print(len(ellipse_pts))
                    circle_mask = cv2.fillConvexPoly(black_bg, ellipse_pts, color=(1))
                    iou = np.sum((tmp_mask*circle_mask)) / (np.sum(tmp_mask + circle_mask)-np.sum((tmp_mask*circle_mask)))
                    # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 5),
                    #                          sharex=True, sharey=True)
                    # axes[0].imshow(circle_mask)
                    # axes[0].set_title('circle_mask')
                    # axes[1].imshow(tmp_mask)
                    # axes[1].set_title('tmp_mask')
                    # axes[2].imshow(img_bgr)
                    # axes[2].set_title('img')
                    # fig.tight_layout()
                    # plt.show()
                    print("iou", iou)
                    ratio = rot_rect[1][0]/rot_rect[1][1]
                    print("ratio", ratio)
                    assert not (rot_rect[1][0] > rot_rect[1][1])
                    if iou > 0.75 and ratio > 0.7 and sy > 5 and sy < 50:
                        """
                        ratio > 0.5 --> close to a circle
                        sy > 50 --> circle is large enough
                        """
                        score = ratio * sy
                        contour_list.append({"contour":contour, "color":k, "ratio":ratio,"radius":(sy+sx)/2, "rot_rect":rot_rect})

                        ## DRAWING
                        # cv.ellipse(img_bgr, rot_rect, (0, 255, 255), 3, cv.LINE_AA)

                        # ellipse_pts = cv.ellipse2Poly((cx, cy), (sx, sy), int(rot_rect[2]), 0, 360, 10)
                        # cv.circle(img_bgr, (cx, cy), 5, (100, 255, 0), 5)
                        # cv.polylines(img_bgr, [ellipse_pts], True, (255, 255, 0))
                        # plt_show(circle_mask)
                        # plt_show(contour_mask)
                        # print(iou)
        # assert len(contour_list) == 1
        score_max = 0
        id_max = -1
        for id, c in enumerate(contour_list):
            if c["ratio"]*c["radius"] > score_max:
                color = c["color"]
                score_max = c["ratio"]*c["radius"]
                id_max =id
        if id_max != -1:
            cv.ellipse(img_bgr, contour_list[id_max]["rot_rect"], (0, 255, 255), 3, cv.LINE_AA)
            print("detected color", contour_list[id_max]["color"])
            return img_bgr, contour_list[id_max]["color"]

        else:
            print("detected color", "none")

            return img_bgr, 'none'


        """ 
        hough circles alg detects all possible arcs that may be extracted from a perfect circle.
        our circle in the image may be skewed due to the camera
        so we use another method for detecting the circles.
        """
        # rows = gray.shape[0]
        # circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
        #                           param1=150,  ## higher threshold of the two passed to Canny
        #                           param2=10,  ## the smaller, the more false circles may be detected
        #                           minRadius=1, maxRadius=30)

        # if circles is not None:
        #     circles = np.uint16(np.around(circles))
        #
        #     if True:
        #         for i in circles[0, :]:
        #             center = (i[0], i[1])
        #             # circle center
        #             cv.circle(img_bgr, center, 1, (0, 100, 100), 3)
        #             # circle outline
        #             radius = i[2]
        #             cv.circle(img_bgr, center, radius, (255, 0, 255), 3)
        #
        #     img = cv.drawContours(img_bgr, contours, -1, (0, 255, 0), 3)
        #     return img
        # else:
        # img = cv.drawContours(img_bgr, contours, -1, (0, 255, 0), 3)

        # return np.expand_dims(gray, axis=-1)

def get_circle_detector():
    return BlobCircleDetector()
    # return CircleDetector()

def main(argv):
    # default_file = 'circle_samples/2019-Dated-U.S.-Coins.jpg'
    # default_file = 'circle_samples/b078wt6js9.main.jpg'
    default_file = 'test_data/210517_141405.962206_c.png'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(filename, cv.IMREAD_UNCHANGED)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1

    circle_detector = CircleDetector()

    circles = circle_detector(src)
    if True:
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 3)

        cv.imwrite("tmp_circle.png", src)
        plt_show(src)

    return 0

if __name__ == "__main__":
    main(sys.argv[1:])