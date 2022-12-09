#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------
# ORT analysis - script video distributor - developed by Victor Ibañez
# 03.04.2021
# -------------------------------------------------------------------------------------


# -----------------------------------------------------
# import libraries
# ----------------------------------------------------- 

import os
import glob
import cv2
import numpy as np

# -----------------------------------------------------
# distribute videos
# ----------------------------------------------------- 

def vid_distributor(project_path, project_name, ref_point, vid_list, time, ttime):
	
	vidcap = cv2.VideoCapture(vid_list[0])
	fps = vidcap.get(cv2.CAP_PROP_FPS)
	vidcap.release()

	target_time = ttime
	frame_list = []
	start = 15
	t_per_vid = target_time/len(vid_list)
	cut_per_min = int((t_per_vid/time)*fps*60)
	f_per_min = int(1*fps*60)

	for vid in vid_list:

		for i in range(1,time+1):

			end = start+cut_per_min
			vidcap = cv2.VideoCapture(vid)
			vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
			success, image = vidcap.read()
			cnt = 1

			while success and cnt <= (end-start):

				res1 = image[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]] # crop frame
				res = cv2.resize(res1, dsize=(300, 300), interpolation=cv2.INTER_CUBIC) # downscale frame
				frame_list.append(res)
				success,image = vidcap.read()
				cnt += 1

			vidcap.release()
			start = start + f_per_min
		
		start = 15

	height, width, depth = frame_list[0].shape
	size = (width,height)
	out = cv2.VideoWriter(os.path.join(project_path,project_name,project_name)+'.MP4',cv2.VideoWriter_fourcc(*'MP4V'), 15, size)
	
	for i in range(len(frame_list)):
		out.write(frame_list[i])
	out.release()
