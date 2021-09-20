#!/usr/bin/env python

import cv2
 
video_file = 'epoch01_front.mkv'
 
cap = cv2.VideoCapture(video_file)

while cap.isOpened():
    ret, img = cap.read()
    print(img.shape)
    if not ret:
        break
    img = cv2.resize(img, dsize=(200, 112))
    img = img[46:,:] / 255.0
    cv2.imshow(video_file, img)
    cv2.waitKey(1)
 
cap.release()
cv2.destroyAllWindows()
