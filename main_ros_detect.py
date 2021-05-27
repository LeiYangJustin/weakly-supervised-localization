#!/usr/bin/env python
# coding: utf-8

import warnings
import sys

import message_filters
import rospy
from cv_bridge import CvBridge, CvBridgeError
import tf as ros_tf
import sensor_msgs.msg
import visualization_msgs.msg
import geometry_msgs.msg
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from tf import transformations
from visualization_msgs.msg import Marker

import helper
from helper import bboxes_extend
from utils import Drawer


warnings.filterwarnings("ignore", message="", category=FutureWarning)
warnings.filterwarnings("ignore", message="deprecated")

import cv2
import numpy as np
from helper import plt_show
from t_detection import get_detector
# from hough_circle_detector import get_circle_detector
from hough_circle_detector_yl import get_circle_detector

#
def get_img_detect(self, data_rgb, data_depth, camera_info):
    # image = cv2.imread('./demo_data/20210106_173228_rgb.png')
    # img_depth = cv2.imread('./demo_data/20210106_173228_depth.png', cv2.IMREAD_UNCHANGED)
    # img_depth = img_depth.astype(np.float32)*0.001
    # print('Step 0, Load RGB image')
    try:
        img_bgr = self.bridge.imgmsg_to_cv2(data_rgb, "bgr8")
        img_depth = self.bridge.imgmsg_to_cv2(data_depth, "32FC1")
    except CvBridgeError as e:
        print(e)
        return False

    print("here we are")
    cir_img, circle_color_str = self.circle_detector(img_bgr)
    # input()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    label, confidence, grasp_info, cat_img = self.detector.run_with_cvimg_input_with_obj_id(self.todetid, img_rgb, img_depth, draw_path=None)
    # label, confidence, grasp_info, cat_img = self.detector.run_with_cvimg_input(img_rgb, img_depth)

    if cat_img is None:
        self.stable_count = 0
        self.label_last = -1
        return

    header = data_rgb.header
    try:
        img_msg = self.bridge.cv2_to_imgmsg(cat_img, "bgr8")
        img_msg.header = header
        self.image_pub.publish(img_msg)

        img_msg = self.bridge.cv2_to_imgmsg(cir_img, "bgr8")
        self.circle_pub.publish(img_msg)
    except CvBridgeError as e:
        print(e)
        return False


    if grasp_info is None or not(confidence == 1.00000):
        self.stable_count = 0
        self.label_last = -1
        return

    if self.label_last == label:
        self.stable_count += 1
    else:
        self.stable_count = 0
    self.label_last = label

    xyz = grasp_info[0]
    if self.stable_count >=2:
        print(f"--------stable: {self.stable_count}, get grasp {xyz}")

        rot_matx = grasp_info[1]
        if np.dot(grasp_info[1], [1,0,0])<0:
            rot_matx = -rot_matx
            print("we are doing it already")

        rot_matz = grasp_info[2]
        if np.dot(grasp_info[2], [0,0,1])<0:
            rot_matz = -rot_matz

        rot_maty = -np.cross(rot_matx,rot_matz)
        # rot_matz = np.cross(rot_matx,rot_maty)
        rot_mat44 = np.eye(4)
        rot_mat44[:3,0] = rot_matx
        rot_mat44[:3,1] = rot_maty
        rot_mat44[:3,2] = rot_matz

        tf_obj2zup = transformations.euler_matrix(*np.deg2rad([180, 0,0]))
        rot_ee = rot_mat44.dot(tf_obj2zup)
        # tf_cam2obj = grasp_info
        # xyz = transformations.translation_from_matrix(tf_cam2obj)
        # quat = transformations.rotation_from_matrix(tf_cam2obj)
        quat = transformations.quaternion_from_matrix(rot_ee)

        self.br.sendTransform(xyz,
                              quat,
                              header.stamp,
                              "obj_{:02d}_o".format(label),
                              "camera_color_optical_frame")

        self.br.sendTransform(xyz,
                              quat,
                              header.stamp,
                              "obj_{:02d}".format(label),
                              "camera")
        msg = PoseStamped()
        msg.header = header
        if circle_color_str == "none":
            msg.header.frame_id = "{}".format(label)
        else:
            msg.header.frame_id = "{}_{}".format(label, circle_color_str)
        msg.pose = helper.gen_pose(xyz[0],xyz[1], xyz[2], quat[0], quat[1],quat[2],quat[3])
        self.object_pub.publish(msg)

    # tf_obj2ee = transformations.euler_matrix(*np.deg2rad([0, 90, 0]))
    # quat_ee = transformations.quaternion_from_matrix(tf_obj2ee)
    # self.br.sendTransform([0,0.3,0],
    #                       quat_ee,
    #                       header.stamp,
    #                       "objee",
    #                       "obj_{:02d}".format(label))
    #
    # tf_obj2ee = transformations.euler_matrix(*np.deg2rad([0, 90, -90]))
    # quat_ee = transformations.quaternion_from_matrix(tf_obj2ee)
    # self.br.sendTransform([0, 0.6, 0],
    #                       quat_ee,
    #                       header.stamp,
    #                       "objee2",
    #                       "obj_{:02d}".format(label))

class image_converter:
    def __init__(self):
        # self.image_rect_pub = rospy.Publisher("~image_rect", sensor_msgs.msg.Image, queue_size=2)
        self.image_pub = rospy.Publisher("~image_detect", sensor_msgs.msg.Image, queue_size=2)
        self.circle_pub = rospy.Publisher("~circle_detect", sensor_msgs.msg.Image, queue_size=2)
        # self.image_3dbbox_pub = rospy.Publisher("~image_detect_3bbox", sensor_msgs.msg.Image, queue_size=2)
        self.object_pub = rospy.Publisher("~obj_pose", geometry_msgs.msg.PoseStamped, queue_size=2)
        self.marker_pub = rospy.Publisher("~obj_marker", visualization_msgs.msg.Marker, queue_size=2)
        self.bridge = CvBridge()

        self.count = 0
        self.first = True
        self.last_key = -1
        self.label_last = -1
        self.stable_count=0
        self.plane=None

        listener = ros_tf.TransformListener()
        # listener.waitForTransform('camera', 'board', rospy.Time(0), rospy.Duration(5))
        # tr_cam2board = listener.lookupTransform('camera', 'board', rospy.Time(0))
        # tf_cam2board = listener.fromTranslationRotation(*tr_cam2board)
        # plane_d = -tf_cam2board[:3,2].dot(tf_cam2board[:3,3])
        # self.plane = np.zeros(4)
        # self.plane[:3]=tf_cam2board[:3,2]
        # self.plane[3] = plane_d

        self.listener = listener
        # self.tf_board2cam = tf_board2cam
        # self.tf_board2obj_z = 0.02
        self.br = ros_tf.TransformBroadcaster()
        self.todetid=0

        self.detector=get_detector(plane=self.plane)
        self.circle_detector=get_circle_detector()

        self.rgb_sub = message_filters.Subscriber('/camera/color/image_raw', sensor_msgs.msg.Image)
        self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', sensor_msgs.msg.Image)
        self.info_sub = message_filters.Subscriber('/camera/color/camera_info', sensor_msgs.msg.CameraInfo)
        self.ts = message_filters.TimeSynchronizer([self.rgb_sub, self.depth_sub, self.info_sub], 1)
        self.ts.registerCallback(self.callback)

        self.chatsub = rospy.Subscriber("chat", String, self.chatcallback)

    def callback(self, rgb, depth, camera_info):
        get_img_detect(self, rgb, depth, camera_info)


    def chatcallback(self, msg:String):
        if msg.data == "0":
            self.todetid = 0
        elif msg.data == "1":
            self.todetid = 1
        elif msg.data == "2":
            self.todetid = 2


def main(args):
    # listener = tf_listener()
    rospy.init_node('obj_detect', anonymous=False)
    ic = image_converter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)