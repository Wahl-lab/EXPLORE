#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------
# ORT analysis - main script prediction - developed by Victor Ibañez
# 03.04.2021
# -------------------------------------------------------------------------------------

# -----------------------------------------------------
# import libraries
# -----------------------------------------------------

import os, glob
import ast
import shutil
import tkinter as tk
from tkinter import simpledialog
from tkinter import filedialog


# -----------------------------------------------------
# create GUI
# -----------------------------------------------------

# gather necessary infos

import GUI_correct

project_path, videos_pred, video_source_path = GUI_correct.GUI_correct()

project_name = os.path.basename(project_path)
label_path = os.path.join(project_path,'labeled')
training_path = os.path.join(project_path, 'training')
plot_path = os.path.join(project_path, 'plots')
vids_source = os.listdir(video_source_path)

ending = vids_source[0].split('.')[-1]

log = open(os.path.join(project_path,"logfile"),"r")
file=log.readlines()
ref_point = ast.literal_eval(file[10])
obj_key = ast.literal_eval(file[16])
time = ast.literal_eval(file[7])
iteration = ast.literal_eval(file[19])
log.close()

with open(os.path.join(project_path,"logfile"),"r") as f:
    contents = f.readlines()

new_iteration = str(iteration+1)+'\n'
contents.remove(str(iteration)+'\n')
contents.insert(19, new_iteration)

with open(os.path.join(project_path,"logfile"),"w") as f:
    contents = "".join(contents)
    f.write(contents)

objects = list(obj_key.keys())
keys = list(obj_key.values())
objects.append('no')
keys.append('n')

import video_scroll_correct
import create_raw_data_correct

dics = video_scroll_correct.video_scroll_corr(videos_pred, objects, keys)

for vid in enumerate(videos_pred):
    vid_name = os.path.basename(vid[1]).split('.')[0]
    vid_path = os.path.join(video_source_path, vid_name + '.' + ending).replace("\\","/")
    create_raw_data_correct.create_raw_data_corr(vid_path, label_path, dics[vid[0]], vid[0], ref_point, time, iteration)
    print('processed data from video:',vid_name)

print('create new training data...')
import create_training_data

create_training_data.create_training_data(label_path, training_path, objects)

print('retrain on new training data...')
import network_multi

network_multi.network_multi(label_path, training_path, os.path.dirname(project_path), project_name, plot_path)

try:
    shutil.rmtree(training_path)
except OSError as e:
    print("Error: %s : %s" % (training_path, e.strerror))

