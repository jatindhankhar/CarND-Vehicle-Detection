import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import pickle 
import cv2
import numpy as np
from lesson_functions import *
from scipy.ndimage.measurements import label

with open('classifier.pickle', 'rb') as handle:                                   
    data = pickle.load(handle) 

svc = data['svc'] 
X_scaler = data['X_scaler']
color_space = data['color_space']
spatial_size = data['spatial_size']
hist_bins = data['hist_bins']
orient = data['orient']
pix_per_cell = data['pix_per_cell']
cell_per_block = data ['cell_per_block']
hog_channel = data['hog_channel']
spatial_feat = data ['spatial_feat']
hist_feat = data['hist_feat']
hog_feat = data['hog_feat']

ystart = 400
ystop = 656
scale = 1.5

heatmaps = []
heatmap_sum = np.zeros((720,1280)).astype(np.float64)


def pipeline(img):
    global heatmaps
    global heatmap_sum
    
    windows,out_img = find_cars(img,ystart,ystop,scale,svc,X_scaler,orient,pix_per_cell,cell_per_block,spatial_size,hist_bins)
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    heat = add_heat(heat,windows)
    heat = apply_threshold(heat,2) 
    draw_image = np.copy(img)
    heatmap_current = np.clip(heat, 0, 255)
    
    heatmap_sum = heatmap_sum + heatmap_current
    heatmaps.append(heat)
    
    # subtract off old heat map to keep running sum of last n heatmaps
    if len(heatmaps)>12:
        old_heatmap = heatmaps.pop(0)
        heatmap_sum -= old_heatmap
        heatmap_sum = np.clip(heatmap_sum,0.0,1000000.0)
    labels = label(heatmap_sum)
    draw_image = draw_labeled_bboxes(draw_image, labels)
    return draw_image


def pipeline_alt(image):
    
    scales = [1, 1.5, 2, 2.5, 4]
    y_start_stops = [[380, 460], [380, 560], [380, 620], [380, 680], [350, 700]]
    
    hot_windows = spot_cars(image, y_start_stops, scales, svc, X_scaler, 
              spatial_size, hist_bins, 
              orient, pix_per_cell, cell_per_block,
              hog_channel, spatial_feat, hist_feat, hog_feat)

    # Read in image similar to one shown above 
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    
    # Add heat to each box in box list
    heat = add_heat(heat,hot_windows)

    heat = apply_threshold(heat,3)    

    
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    
    # draw the bounding box on the image 
    draw_image = np.copy(image)
    draw_image = draw_labeled_bboxes(draw_image, labels)
    
    return draw_image


from moviepy.editor import VideoFileClip
from IPython.display import HTML


output_file = 'project_video_processed.mp4'
input_clip = VideoFileClip("project_video.mp4")
output_clip = input_clip.fl_image(pipeline)
%time output_clip.write_videofile(output_file, audio=False) 
