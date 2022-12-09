
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------
# ORT analysis - script prediction on multiple classes - developed by Victor Ibañez
# 03.04.2021
# -------------------------------------------------------------------------------------


# -----------------------------------------------------
# import libraries
# ----------------------------------------------------- 

import cv2
import numpy as np
from scipy import stats
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import array_to_img
import math
import os, glob
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.InteractiveSession(config=config)

# On mac you need to shut this down
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def predict_frames_multi(project_path, model, videos, time, ref_point, names, objects, bins, o_coord):

	# import model
	project_name = os.path.basename(project_path)
	loaded_model = tf.keras.models.load_model(model)

	data = []
	names.append('no')
	names.sort()
	index = names.index('no')
	color_list_fill = [(255,102,255),(102,255,102),(102,255,255),(102,102,255),(107,178,255),(255,102,102)]
	color_list_box = [(255,0,255),(0,255,0),(0,255,255),(0,0,255),(0,128,255),(255,0,0)]

	# import video
	def frame_prediction(video, time, vid_name, ref_point):
		vidcap = cv2.VideoCapture(video)
		success,frame = vidcap.read()
		fps =  vidcap.get(cv2.CAP_PROP_FPS)

		count = 1
		max_time = (int(time)*60*int(fps))+15

		freq_list = []
		cnt_list = []
		for i in range(len(names)):
			cnt_list.append(0)
			freq_list.append(0)

		create_list = []

		image_list = []

		while success:

			if count > 15 and count < max_time:

				res1 = frame[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]] # crop video
				res = cv2.resize(res1, dsize=(150, 150), interpolation=cv2.INTER_CUBIC)
				i = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
				i = i/255.
				image_list.append(i)

				res_gray = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
				res_col = cv2.cvtColor(res_gray, cv2.COLOR_GRAY2BGR)
				create_list.append(res_col)


			elif count >= max_time:
				break

			success,frame = vidcap.read()
			count += 1

		img_array = np.array(image_list)
		result = loaded_model.predict_classes(img_array)
		result1 = loaded_model.predict_proba(img_array)

		cnt_min = 1

		for i in range(len(result)):

			if i > 0:
				if result[i] != index and result[i] != result[i-1]:
					freq_list[int(result[i])] += 1

			if len(names) > 2:
				prob_result = result1[i][result[i]]
			else:
				prob_result = result1[i][0]

			if result[i] != index and prob_result > 0.99:

				cnt_list[int(result[i])] += 1

				res_col = create_list[i]
				for j in range(len(objects)):
					if objects[j] == names[int(result[i])]:
						overlay = res_col.copy()
						overlay[o_coord[j][0][1]:o_coord[j][1][1], o_coord[j][0][0]:o_coord[j][1][0]] = color_list_fill[j]
						o = cv2.addWeighted(overlay,0.5,res_col,0.5,0)
						res_col = o
						cv2.rectangle(res_col, (o_coord[j][0][0], o_coord[j][0][1]), (o_coord[j][1][0], o_coord[j][1][1]), color_list_box[j], thickness = 2)
				cv2.putText(res_col, names[int(result[i])], (10, 30), cv2.FONT_HERSHEY_SIMPLEX , 1,(255, 255, 255), 3, cv2.LINE_AA, False)
				create_list[i] = res_col

			if (i+2) % (int(fps)*60*int(bins)) == 0:

				d = {'experiment':os.path.basename(os.path.dirname(videos[0])),'animal':vid_name,'minute':cnt_min*int(bins)}

				for j in range(len(names)):
					if j == index:
						continue
					else:
						d[names[j]] = cnt_list[j]/int(fps)
						d[names[j]+'_freq'] = freq_list[j]

				print('minute:',cnt_min*int(bins))

				data.append(d)
				print(d)
				cnt_min += 1
				for j in range(len(cnt_list)):
					cnt_list[j] = 0
				for j in range(len(freq_list)):
					freq_list[j] = 0

		height, width, layers = res_col.shape
		size = (width,height)
		name = os.path.join(project_path,'prediction_videos',vid_name)
		out = cv2.VideoWriter(name + '.MP4', cv2.VideoWriter_fourcc(*'MP4V'), 25, size)

		for i in create_list:
			out.write(i)
		out.release()

		print('Done!', count, 'frames predicted from video:', video)

	print('start predicting...')

	for video in videos:
		split = video.split('.')
		vid_name = os.path.basename(split[0])
		print('processing video:', vid_name)
		frame_prediction(video, time, vid_name, ref_point)

	print('all videos processed!')

	# write a csv with all values
	exp_name = os.path.join(project_path,project_name)
	df = pd.DataFrame.from_dict(data)
	filename = exp_name + '.csv'

	if os.path.isfile(filename):
		cnt = 1
		while True:
			new_filename = exp_name + '_' + str(cnt) + '.csv'
			cnt += 1
			if os.path.isfile(new_filename):
				continue
			else:
				filename = new_filename
				break
	df.to_csv(filename, index=False)


