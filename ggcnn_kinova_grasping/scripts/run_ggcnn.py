#! /usr/bin/env python

import time
import os
import copy
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import tensorflow as tf
from keras.models import load_model

import cv2
import scipy.ndimage as ndimage
import skimage.draw
from skimage.draw import circle
from skimage.feature import peak_local_max

import rospy
import tf2_ros
import geometry_msgs.msg
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray

import spartan.utils.utils as spartan_utils
from director.thirdparty import transformations



bridge = CvBridge()

# Load the Network.

spartan_source_dir = spartan_utils.getSpartanSourceDir()
model_rel_to_spartan_source = 'modules/spartan/ggcnn/data/networks/ggcnn_rss/epoch_29_model.hdf5'
MODEL_FILE = os.path.join(spartan_source_dir, model_rel_to_spartan_source)
# MODEL_FILE = 'PATH/TO/model.hdf5'
model = load_model(MODEL_FILE)

rospy.init_node('ggcnn_detection')

# Output publishers.
grasp_pub = rospy.Publisher('ggcnn/img/grasp', Image, queue_size=1)
grasp_line_pub = rospy.Publisher('ggcnn/img/grasp_line', Image, queue_size=1)
grasp_plain_pub = rospy.Publisher('ggcnn/img/grasp_plain', Image, queue_size=1)
depth_pub = rospy.Publisher('ggcnn/img/depth', Image, queue_size=1)
ang_pub = rospy.Publisher('ggcnn/img/ang', Image, queue_size=1)
cmd_pub = rospy.Publisher('ggcnn/out/command', Float32MultiArray, queue_size=1)

tf2_broadcaster = tf2_ros.TransformBroadcaster()

# Initialise some globals.
prev_mp = np.array([150, 150])
ROBOT_Z = 0
ROBOT_Z = 0.5 # manuelli: hack for now
ALWAYS_MAX = True  # Use ALWAYS_MAX = True for the open-loop solution.
NETWORK_IMG_SIZE = 300
WIDTH_SCALE_FACTOR = 150

# Tensorflow graph to allow use in callback.
graph = tf.get_default_graph()

cameraName = 'camera_carmine_1'
pointCloudTopic = '/' + str(cameraName) + '/depth/points'
rgbImageTopic   = '/' + str(cameraName) + '/rgb/image_rect_color'
depthImageTopic = '/' + str(cameraName) + '/depth_registered/sw_registered/image_rect'
camera_info_topic = '/' + str(cameraName) + '/rgb/camera_info'
graspFrameName = 'base'
depthOpticalFrameName = cameraName + "_depth_optical_frame"
rgbOpticalFrameName = cameraName + "_rgb_optical_frame"

# Get the camera parameters
rospy.loginfo("Wating for CameraInfo msg . . .")
camera_info_msg = rospy.wait_for_message(camera_info_topic, CameraInfo)
rospy.loginfo("received for CameraInfo msg")
K = camera_info_msg.K
fx = K[0]
cx = K[2]
fy = K[4]
cy = K[5]
image_width = camera_info_msg.width
image_height = camera_info_msg.height


# Execution Timing
class TimeIt:
    def __init__(self, s):
        self.s = s
        self.t0 = None
        self.t1 = None
        self.print_output = False

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, t, value, traceback):
        self.t1 = time.time()
        if self.print_output:
            print('%s: %s' % (self.s, self.t1 - self.t0))




def robot_pos_callback(data):
    global ROBOT_Z
    ROBOT_Z = data.pose.position.z


def depth_callback(depth_message):
    rospy.loginfo("received a depth message")
    global model
    global graph
    global prev_mp
    global ROBOT_Z
    global fx, cx, fy, cy

    print "type(depth_message):", type(depth_message)

    with TimeIt('Crop'):
        depth = bridge.imgmsg_to_cv2(depth_message)
        print "type(depth):", type(depth)
        print "depth.dtype:", depth.dtype
        print "depth min", np.min(depth)
        print "depth max", np.max(depth)
        print "depth.shape", depth.shape
        print "depth[200,200]", depth[200, 200]

        # Crop a square out of the middle of the depth and resize it to 300*300
        crop_size = 400
        depth_crop = cv2.resize(depth[(image_height-crop_size)//2:(image_height-crop_size)//2+crop_size, (image_width-crop_size)//2:(image_width-crop_size)//2+crop_size], (300, 300))

        # Replace nan with 0 for inpainting.
        depth_crop = depth_crop.copy()
        depth_nan = np.isnan(depth_crop).copy()
        depth_crop[depth_nan] = 0 # set nan's to zero

    with TimeIt('Inpaint'):
        # open cv inpainting does weird things at the border.
        depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)

        mask = (depth_crop == 0).astype(np.uint8)
        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        depth_scale = np.abs(depth_crop).max()
        depth_crop = depth_crop.astype(np.float32)/depth_scale  # Has to be float32, 64 not supported.

        depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS)

        # Back to original size and value range.
        depth_crop = depth_crop[1:-1, 1:-1]
        depth_crop = depth_crop * depth_scale

    with TimeIt('Calculate Depth'):
        # Figure out roughly the depth in mm of the part between the grippers for collision avoidance.
        depth_center = depth_crop[100:141, 130:171].flatten()
        depth_center.sort()
        depth_center = depth_center[:10].mean() * 1000.0

    with TimeIt('Inference'):
        # Run it through the network.
        depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1)
        with graph.as_default():
            pred_out = model.predict(depth_crop.reshape((1, 300, 300, 1)))

        points_out = pred_out[0].squeeze()
        points_out[depth_nan] = 0

    with TimeIt('Trig'):
        # Calculate the angle map.
        cos_out = pred_out[1].squeeze()
        sin_out = pred_out[2].squeeze()
        ang_out = np.arctan2(sin_out, cos_out)/2.0

        width_out = pred_out[3].squeeze() * 150.0  # Scaled 0-150:0-1

    with TimeIt('Filter'):
        # Filter the outputs.
        points_out = ndimage.filters.gaussian_filter(points_out, 5.0)  # 3.0
        ang_out = ndimage.filters.gaussian_filter(ang_out, 2.0)

    with TimeIt('Control'):
        # Calculate the best pose from the camera intrinsics.
        maxes = None

        

        if ROBOT_Z > 0.34 or ALWAYS_MAX:  # > 0.34 initialises the max tracking when the robot is reset.
            # Track the global max.
            max_pixel = np.array(np.unravel_index(np.argmax(points_out), points_out.shape))
            prev_mp = max_pixel.astype(np.int)
        else:
            # Calculate a set of local maxes.  Choose the one that is closes to the previous one.
            maxes = peak_local_max(points_out, min_distance=10, threshold_abs=0.1, num_peaks=3)
            if maxes.shape[0] == 0:
                return
            max_pixel = maxes[np.argmin(np.linalg.norm(maxes - prev_mp, axis=1))]

            # Keep a global copy for next iteration.
            prev_mp = (max_pixel * 0.25 + prev_mp * 0.75).astype(np.int)

        # note: prev_mp is in the depth_crop image coordinates, so it's the
        # [300, 300] format
        max_pixel_crop_coords = copy.copy(prev_mp)
        ang = ang_out[max_pixel[0], max_pixel[1]]
        width = width_out[max_pixel[0], max_pixel[1]]

        # Convert max_pixel back to uncropped/resized image coordinates in order to do the camera transform.
        max_pixel = ((np.array(max_pixel) / 300.0 * crop_size) + np.array([(image_height - crop_size)//2, (image_width - crop_size) // 2]))
        max_pixel = np.round(max_pixel).astype(np.int)

        # now max_pixel is now the original image coordinates [480, 640]

        point_depth = depth[max_pixel[0], max_pixel[1]]

        # These magic numbers are my camera intrinsic parameters.
        x = (max_pixel[1] - cx)/(fx) * point_depth
        y = (max_pixel[0] - cy)/(fy) * point_depth
        z = point_depth

        if np.isnan(z):
            return False


        print "x,y,z: ", [x,y,z]

    with TimeIt('Draw'):
        # Draw grasp markers on the points_out and publish it. (for visualisation)
        grasp_img = np.zeros((300, 300, 3), dtype=np.uint8)
        grasp_img[:,:,2] = (points_out * 255.0)

        grasp_img_plain = grasp_img.copy()
        grasp_line_img = grasp_img.copy()

        rr, cc = circle(prev_mp[0], prev_mp[1], 5)

        grasp_img[rr, cc, 0] = 0
        grasp_img[rr, cc, 1] = 255
        grasp_img[rr, cc, 2] = 0


        # need start and end pixels of line
        width_in_pixels = width*WIDTH_SCALE_FACTOR

        # print "ang", ang
        print "ang (deg):", np.rad2deg(ang)
        # print "max_pixel_crop_coords", max_pixel_crop_coords
        angle_direction_in_img = np.array([-np.sin(ang), np.cos(ang)])

        # note left/right have no real meaning here
        grasp_left_finger = max_pixel_crop_coords + width/2.0*angle_direction_in_img
        grasp_right_finger = max_pixel_crop_coords - width/2.0*angle_direction_in_img

        grasp_left_finger = np.round(grasp_left_finger).astype(np.int)
        grasp_right_finger = np.round(grasp_right_finger).astype(np.int)

        

        # clip to pixel coords
        grasp_left_finger = np.clip(grasp_left_finger, 0, NETWORK_IMG_SIZE - 1)
        grasp_right_finger = np.clip(grasp_right_finger, 0, NETWORK_IMG_SIZE - 1)

        


        # open CV has (u,v) format rather than (row, col)

        cv2.line(grasp_line_img, tuple(grasp_left_finger[::-1]), tuple(grasp_right_finger[::-1]), (255,0,0), thickness=5)

        cv2.circle(grasp_line_img, tuple(max_pixel_crop_coords[::-1]), 5, (0, 255, 0), -1)

        # rr_line, cc_line = skimage.draw.line(grasp_left_finger[0], grasp_left_finger[1], grasp_right_finger[0], grasp_right_finger[1])
        #
        # grasp_line_img[rr_line, cc_line, :] = np.array([255, 0, 0]) # should be blue
        # grasp_line_img[rr, cc, :] = np.array([0,255,0]) # green: grasp center




    with TimeIt('Publish'):
        # Publish the output images (not used for control, only visualisation)
        grasp_img = bridge.cv2_to_imgmsg(grasp_img, 'bgr8')
        grasp_img.header = depth_message.header
        grasp_pub.publish(grasp_img)

        grasp_line_img_msg = bridge.cv2_to_imgmsg(grasp_line_img, 'bgr8')
        grasp_line_img_msg.header = depth_message.header
        grasp_line_pub.publish(grasp_line_img_msg)

        grasp_img_plain = bridge.cv2_to_imgmsg(grasp_img_plain, 'bgr8')
        grasp_img_plain.header = depth_message.header
        grasp_plain_pub.publish(grasp_img_plain)

        depth_pub.publish(bridge.cv2_to_imgmsg(depth_crop))

        ang_pub.publish(bridge.cv2_to_imgmsg(ang_out))

        # Output the best grasp pose relative to camera.
        cmd_msg = Float32MultiArray()
        cmd_msg.data = [x, y, z, ang, width, depth_center]
        cmd_pub.publish(cmd_msg)

        # grasp to camera frame transform
        grasp_to_camera = geometry_msgs.msg.TransformStamped()
        grasp_to_camera.header = depth_message.header
        grasp_to_camera.header.frame_id = rgbOpticalFrameName
        grasp_to_camera.child_frame_id = "ggcnn_grasp"

        grasp_to_camera.transform.translation.x = x
        grasp_to_camera.transform.translation.y = y
        grasp_to_camera.transform.translation.z = z

        quat_wxyz = transformations.quaternion_about_axis(ang, [0,0,1])
        grasp_to_camera.transform.rotation.w = quat_wxyz[0]
        grasp_to_camera.transform.rotation.x = quat_wxyz[1]
        grasp_to_camera.transform.rotation.y = quat_wxyz[2]
        grasp_to_camera.transform.rotation.z = quat_wxyz[3]


        tf2_broadcaster.sendTransform(grasp_to_camera)



    

# def visualize_grasp(grasp_quality_img, depth_crop, ang_img, width_img,
#                     depth_img, max_pixel, max_pixel_crop):
#     # Draw grasp markers on the points_out and publish it. (for visualisation)
#     grasp_img = np.zeros((NETWORK_IMG_SIZE, NETWORK_IMG_SIZE, 3), dtype=np.uint8)
#     grasp_img[:, :, 2] = (grasp_quality_img * 255.0)
#
#     grasp_img_plain = grasp_img.copy()
#
#     rr, cc = circle(prev_mp[0], prev_mp[1], 5)
#     grasp_img[rr, cc, 0] = 0
#     grasp_img[rr, cc, 1] = 255
#     grasp_img[rr, cc, 2] = 0
#
#
#     with TimeIt('Publish'):
#         # Publish the output images (not used for control, only visualisation)
#         grasp_img = bridge.cv2_to_imgmsg(grasp_img, 'bgr8')
#         grasp_img.header = depth_message.header
#         grasp_pub.publish(grasp_img)
#
#         grasp_img_plain = bridge.cv2_to_imgmsg(grasp_img_plain, 'bgr8')
#         grasp_img_plain.header = depth_message.header
#         grasp_plain_pub.publish(grasp_img_plain)
#
#         depth_pub.publish(bridge.cv2_to_imgmsg(depth_crop))
#
#         ang_pub.publish(bridge.cv2_to_imgmsg(ang_out))
#
#         # Output the best grasp pose relative to camera.
#         cmd_msg = Float32MultiArray()
#         cmd_msg.data = [x, y, z, ang, width, depth_center]
#         cmd_pub.publish(cmd_msg)
#
#
#     """
#     Visualizes the grasp
#     :return:
#     :rtype:
#     """


# depth_sub = rospy.Subscriber('/camera/depth/image_meters', Image, depth_callback, queue_size=1)

# the depth image should be in meters
depth_sub = rospy.Subscriber(depthImageTopic, Image, depth_callback, queue_size=1)
# robot_pos_sub = rospy.Subscriber('/m1n6s200_driver/out/tool_pose', PoseStamped, robot_pos_callback, queue_size=1)

while not rospy.is_shutdown():
    rospy.spin()
