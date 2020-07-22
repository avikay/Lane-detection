# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 20:22:18 2020

@author: Avinash
"""

"""
Detecting lines in video
"""

import cv2
import numpy as np

video = cv2.VideoCapture("road.mp4")

while True:
    ret, frame1 = video.read()
    frame = cv2.GaussianBlur(frame1, (5,5),0)
    #Since we need the video is small we are restarting the video every time it ends using the ret value 
    if not ret:
        video = cv2.VideoCapture("road.mp4")
        continue
    #We can use HSV value for edges detection as well but here 
    #Gray scale works much better than the HSV masked frames
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    low_white = np.array([0, 0, 168], dtype = np.uint8)
    high_white = np.array([172, 111, 255], dtype = np.uint8)
    
    mask = cv2.inRange(hsv, low_white, high_white)
    
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(mask, 75, 150)
    
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap = 350)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1,y1), (x2,y2), (0,0,255), 5)
    
    
    cv2.imshow("frame",frame)
    cv2.imshow("Edges", edges)
    
    key = cv2.waitKey(25)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()