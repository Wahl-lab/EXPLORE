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


import GUI_manual_scoring

project_path, project_name, time, videos, objects, o_keys = GUI_manual_scoring.GUI_manual_scoring()

objects[:] = [x for x in objects if x]
o_keys[:] = [x for x in o_keys if x]


time = int(time)

project_folder = os.path.join(project_path, project_name)

if not os.path.isdir(project_folder): 
    os.makedirs(project_folder)



# -----------------------------------------------------
# hand label
# -----------------------------------------------------


names = objects
keys = o_keys

# hand label video
import hand_label

hand_label.hand_label(project_folder, videos, time, names, keys)






