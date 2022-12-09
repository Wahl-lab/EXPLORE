#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------
# ORT analysis - main script training - developed by Victor Ibañez
# 03.04.2021
# -------------------------------------------------------------------------------------


# -----------------------------------------------------
# import libraries
# -----------------------------------------------------


import os, glob
import shutil
import random
import tkinter as tk
from tkinter import simpledialog
from tkinter import filedialog


# -----------------------------------------------------
# define project settings
# -----------------------------------------------------


import GUI_training

project_path, project_name, time, selection, videos, objects, o_keys, nk, ttime = GUI_training.GUI_training()

objects[:] = [x for x in objects if x]
o_keys[:] = [x for x in o_keys if x]

time = int(time)
ttime = int(ttime)

project_folder = os.path.join(project_path, project_name)

if not os.path.isdir(project_folder): 
    os.makedirs(project_folder)

if selection == 1:
    print('apply kmeans clustering...')
    vid_path = videos

    nk = int(float(nk))

    import random_dist
    videos = random_dist.random_dist(vid_path, nk)

    rs = []
    for i in videos:
        rs.append(os.path.basename(i))
    print('videos randomly sampled: ', rs)


# -----------------------------------------------------
# extract training frames
# -----------------------------------------------------


# crop frames
import GUI_crop_frames

ref_point = GUI_crop_frames.main(videos[0]) 

import object_coord

coord = []
for i in objects:
    c = object_coord.main(videos[0], ref_point, i)
    coord.append(c)

# create sampled video
print('creating video for labeling...')
import vid_distributor

vid_distributor.vid_distributor(project_path, project_name, ref_point, videos, time, ttime)


# define label paths
label_path = os.path.join(project_folder, 'labeled')

if not os.path.isdir(label_path): 
    os.makedirs(label_path)

names = objects#+behaviors
keys = o_keys#+b_keys

obj_key = {}

for i in range(len(names)):
    obj_key[names[i]]=keys[i]

for i in names:
    path = os.path.join(label_path, i)
    if not os.path.isdir(path): 
        os.makedirs(path)

no_path = os.path.join(label_path, 'no')

if not os.path.isdir(no_path): 
    os.makedirs(no_path)


vid = os.path.join(project_path,project_name,project_name)+'.MP4'


# hand label video
import video_scroll

dic = video_scroll.video_scroll(vid, names, keys)

v_list = []
for i in videos:
    v_list.append(os.path.basename(i))

# write logfile
L = ['project: \n',project_name,'\n','\n',
'sampled videos: \n',str(v_list),'\n','\n',
'video duration: \n',str(time),'\n','\n',
'cropping coordinates: \n',str(ref_point),'\n','\n',
'object coordinates: \n',str(coord),'\n','\n',
'objects: \n',str(obj_key),'\n','\n',
'iteration: \n',str(1),'\n','\n',
'trained frames: \n',str(dic)]

f = open(os.path.join(project_folder,'logfile'), "w+")
f.writelines(L) 
f.close()


# create raw data
print('create raw data...')
import create_raw_data

create_raw_data.create_raw_data(vid, label_path, dic)


# define training and plot paths
training_path = os.path.join(project_folder, 'training')
plot_path = os.path.join(project_folder, 'plots')

if not os.path.isdir(training_path): 
    os.makedirs(training_path)

if not os.path.isdir(plot_path): 
    os.makedirs(plot_path)

# create training data
print('create training data...')
import create_training_data

create_training_data.create_training_data(label_path, training_path, names)


# -----------------------------------------------------
# train the network
# -----------------------------------------------------


import network_multi

network_multi.network_multi(label_path, training_path, project_path, project_name, plot_path)

# delete label and training data
#try:
    #shutil.rmtree(label_path)
#except OSError as e:
    #print("Error: %s : %s" % (label_path, e.strerror))

try:
    shutil.rmtree(training_path)
except OSError as e:
    print("Error: %s : %s" % (training_path, e.strerror))

