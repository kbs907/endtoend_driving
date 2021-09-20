#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy, cv2, time
from sensor_msgs.msg import Image
from xycar_motor.msg import xycar_motor
from cv_bridge import CvBridge

import torch
import torch.nn as nn
import torch.optim as optim

import rospkg
from model import end2end

import glob, random, time, io, dill, os, cv2

import numpy as np

r = rospy.Rate(5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = end2end().to(device)
net = study_model_load(430, 68, net, device)

def camera_callback(data) :
	global cv_image
	global bridge
	cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

def study_model_load(epoch, batch_cnt, model, device):
    rospack = rospkg.RosPack()
    LoadPath_main = rospack.get_path('e2e_drive')+"/save/main_model_"+str(epoch).zfill(6)+"_"+str(batch_cnt).zfill(6)+".pth"
    with open(LoadPath_main, 'rb') as f :
        LoadBuffer = io.BytesIO(f.read())
        model.load_state_dict(torch.load(LoadBuffer, map_location=device))
    return model


def e2e_drive(frame) :
    frame = cv2.resize(frame, (200, 150))
    frame = frame[84:,:]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray,(5, 5), 0)
    edge_img = cv2.Canny(np.uint8(blur_gray), 50, 150)
    frame = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    frame = frame.transpose((2,0,1)) / 255.0
    t_frame = torch.FloatTensor([frame]).to(device)

   	angle = net(t_frame)
	motor_control.speed = 40
   	motor_control.angle = int(round(angle.tolist()[0][0]))
   	pub.publish(motor_control)

   	r.sleep()
	  


def main() :
	cv_image = np.empty(shape=[0])
	bridge = CvBridge()
	rospy.init_node('e2e', anonymous = True)
	rospy.Subscriber("/usb_cam/image_raw/", Image, camera_callback)
	pub = rospy.Publisher("/xycar_motor", xycar_motor, queue_size = 1)

	motor_msg = xycar_motor()
	motor_msg.speed = 30
	r = rospy.Rate(5)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	net = end2end().to(device)
	net = study_model_load(430, 68, net, device)

    while True :
		if cv_image.size != (640 * 480 * 3) :
			continue
		break


	while not rospy.is_shutdown() :
    	start_time = time.time()
    	frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2YUV)
    	#frame = cv2.resize(img, (200, 112))
    	frame = cv2.resize(frame, (200, 150))
    	#frame = frame[46:,:]
    	frame = frame[84:,:]
    	frame = frame.transpose((2,0,1)) / 255.0
    	t_frame = torch.FloatTensor([frame]).to(device)

    	angle = net(t_frame)
    	print(time.time() - start_time)
    	motor_msg.angle = int(round(angle.tolist()[0][0]))
    	print(motor_msg)
    	pub.publish(motor_msg)

    	cv2.imshow('original', cv_image)

    	cv2.waitKey(1)
    	r.sleep()
	  
	cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
