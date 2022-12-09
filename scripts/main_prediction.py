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
import tkinter as tk
from tkinter import simpledialog
from tkinter import filedialog


# -----------------------------------------------------
# create GUI
# -----------------------------------------------------

# gather necessary infos

import GUI_prediction

project_path, model, videos, time, bins = GUI_prediction.GUI_prediction()

time = int(time)

#if create == 1:
vid_path = os.path.join(project_path,'prediction_videos')
if not os.path.isdir(vid_path):
    os.makedirs(vid_path)


log = open(os.path.join(project_path,"logfile"),"r")
file=log.readlines()
ref_point = ast.literal_eval(file[10])
coord = ast.literal_eval(file[13])
obj_key = ast.literal_eval(file[16])
dic = ast.literal_eval(file[22])
log.close()

names = list(dic.keys())
objects = list(obj_key.keys())

# predict on videos

import predict_frames_multi

predict_frames_multi.predict_frames_multi(project_path, model, videos, time, ref_point, names, objects, bins, coord)


