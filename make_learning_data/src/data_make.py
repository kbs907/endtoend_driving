#!/usr/bin/env python

import rospy, cv2, csv, time
import numpy as np
from xycar_msgs.msg import xycar_motor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

name = "data_make"
video_name = "/home/nvidia/Desktop/test.mkv"
csv_name = "/home/nvidia/Desktop/test.csv"

cv_image = np.empty(shape=[0])
bridge = CvBridge()
current_angle = 9999999

def motor_callback(data) :
	global current_angle
	current_angle = data.angle

def camera_callback(data) :
	global cv_image
	global bridge
	cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

rospy.init_node(name, anonymous = True)
rospy.Subscriber("/usb_cam/image_raw/", Image, camera_callback)
rospy.Subscriber("/xycar_motor", xycar_motor, motor_callback)

out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640, 480))

f = open(csv_name, 'w')
wr = csv.writer(f)
wr.writerow(["ts_micro", "frame_index", "wheel"])

rate = rospy.Rate(10)
cnt = 0

while True :
	#if current_angle == 9999999 :
	#	continue
	if cv_image.size != (640 * 480 * 3) :
		continue
	break

while not rospy.is_shutdown() :
	cv2.imshow("haha", cv_image)
	cv2.waitKey(1)
	wr.writerow([time.time(), cnt, current_angle])
	out.write(cv_image)
	rate.sleep()

	if cnt == 30000 :
		break
	cnt += 1

out.release()
f.close()
