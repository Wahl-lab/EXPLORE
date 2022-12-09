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


import GUI_quadrant

project_path, project_name, videos, time, background  = GUI_quadrant.GUI_quadrant() #, plot

time = int(time)

project_folder = os.path.join(project_path, project_name)

if not os.path.isdir(project_folder): 
    os.makedirs(project_folder)

import GUI_crop_frames

ref_point = GUI_crop_frames.main(videos[0])

import quadrant_analysis

quadrant_analysis.quadrant_analysis(project_path, project_name, videos, time, background, ref_point) # plot,




