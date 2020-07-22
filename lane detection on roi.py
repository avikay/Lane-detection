# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 16:22:07 2020

Lane detection using the ROI in an Image

@author: Avinash
"""
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import cv2
import numpy as np
import math
from moviepy.editor import VideoFileClip
from IPython.display import HTML




#Cropping the image for the region of interest
#We are cropping the image by using a mask of exact shape as that of image
def region_of_interest(img, vertices):
    
    mask = np.zeros_like(img)
 
    color_as_imgmask = 255 
    cv2.fillPoly(mask, vertices, color_as_imgmask)
    masked = cv2.bitwise_and(img,mask)
    
    return masked

#Function to draw lines on the lines detected using hough transform
def draw_lane_lines(img, lines, color = [255,0,0], thickness = 5):
   
    
    line_img = np.zeros((img.shape[0],img.shape[1],3), dtype = np.uint8)
    img = np.copy(img)
    
    if lines is None:
        return
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    
    return img

def pipeline(image):
    #Defining vertices for the region of interest   
    height = image.shape[0]
    width = image.shape[1]
    
    vertices_for_roi = [(0,height),
                        (width/2,height/2),
                        (width, height)]
    
    #Generating gray scale image
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #Detecting edges in the image
    edges = cv2.Canny(img_gray, 50,100)
    
    
    
    #Cropping image with edges using region of interest function
    lane_lines = region_of_interest(edges, np.array([vertices_for_roi], np.int32))
    
    
    #Applying Hough Transform for detecting lines
    lines = cv2.HoughLinesP(lane_lines, 1, np.pi/180, 25, maxLineGap = 300)
    
    
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
    
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2-y1)/(x2-x1)
            
            if math.fabs(slope) < 0.5:
                continue
            
            if slope <= 0:
                left_line_x.extend([x1,x2])
                left_line_y.extend([y1,y2])
            elif slope >= 0:
                right_line_x.extend([x1,x2])
                right_line_y.extend([y1,y2])
    
    
    min_y = int(image.shape[0] * (3/5))
    max_y = int(image.shape[0])
    
    poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg = 1.0))
    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))
    
    poly_right = np.poly1d(np.polyfit(right_line_x, right_line_y, deg = 1.0))
    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))
    
    lane_image = draw_lane_lines(image, [[
            [left_x_start, max_y, left_x_end, min_y],
            [right_x_start, max_y, right_x_end, min_y]
        ]])
    return lane_image

 
white_output = 'video_out.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
lane_clip = clip1.fl_image(pipeline)
lane_clip.write_videofile(white_output, audio = False)

