import numpy as np
import cv2

#Interpolation options for frame rescaling
    #INTER_NEAREST - nearest neighbor interpolation
    #INTER_LINEAR - a bilinear interpolation (default option)
    #INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
    #INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood
    #INTER_AREA - resampling using pixel area relation
inter = cv2.INTER_CUBIC
frame_h = 84
frame_w = 84

#Preprocessing for single frame,using information from previous frame
#Input: 2 frames of shape(210,160,3)
def preprocess_frame(curr_state, prev_state):

    #remove flickering between two frames
    curr_state = np.maximum(curr_state, prev_state)
    
    #remove X and Z channels
    curr_state = curr_state[:,:,1]

    #Rescale 210x160 to 84x84
    curr_state = cv2.resize(curr_state, (frame_h,frame_w), interpolation=inter)
    return curr_state
  
def preprocess_first_frame(curr_state):
    #remove X and Z channels
    curr_state = curr_state[:,:,1]

    #Rescale 210x160 to 84x84
    curr_state = cv2.resize(curr_state, (frame_h,frame_w), interpolation=inter)
    return curr_state
