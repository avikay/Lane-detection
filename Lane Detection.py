# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 19:48:59 2020

@author: Avinash
"""

"""
Detecting Lines and Circles in an image
"""

import cv2
import numpy as np

#Reading lines image and converting to gray scale
img_line = cv2.imread("Road.jpg")
img_gray_line = cv2.cvtColor(img_line, cv2.COLOR_BGR2GRAY)

#Reading image with circle and then converting it to gray scale
img_circle = cv2.imread("circle.jpg")
img_gray_circle = cv2.cvtColor(img_circle, cv2.COLOR_BGR2GRAY)

#Edges for lines
edges_line = cv2.Canny(img_gray_line, 250, 250)

#Edges for circle
edges_circle = cv2.Canny(img_gray_circle, 50, 50)

#Applying Hough Transform for lines and circles
lines = cv2.HoughLinesP(edges_line, 1, np.pi/180, 50, maxLineGap = 150)
circle = cv2.HoughCircles(edges_circle, cv2.HOUGH_GRADIENT, 1, 80, param1 = 50, param2= 30, minRadius=0, maxRadius=0)

#Showing lines on the actual image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img_line, (x1, y1), (x2, y2), (0,255,0), 2)


#Showing circle after converting circle to uint16 type
circle = np.uint16(np.around(circle))
for i in circle[0,:]:
    cv2.circle(img_circle, (i[0],i[1]), i[2], (0,255,0),2)
    cv2.circle(img_circle, (i[0],i[1]), 2, (0,0,255), 3)

#Display results
cv2.imshow("Line", img_line)
cv2.imshow("Circle", img_circle)

cv2.waitKey(0)
cv2.destroyAllWindows()
