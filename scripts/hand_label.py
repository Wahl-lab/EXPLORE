import cv2
import numpy as np
import os, glob
import pandas as pd


def hand_label(project_path, videos, time, names, keys):

	data = []
	frames = []
	
	def write(o, i):
		if len(o)==1:
			return 0
		else:
			diff2 = 0
			diff_list = []

			for k in range(0, len(o),2):
				if k+1 >= len(o):
					break
				if i > o[k] and i < o[k+1]:
					diff1 = (i-o[k])
					diff_list.append(diff1)
					diff2 = (o[k+1]-i)

				elif o[k] < i and o[k+1] < i and o[k] > (i-int(fps*60)) and o[k+1] > (i-int(fps*60)):
					diff = o[k+1]-o[k]
					diff_list.append(diff)
					diff2 = 0

			tot = sum(diff_list)/fps
			diff2 = diff2/fps
			return tot, diff2


	for vid in videos:
	
		dic = {}
		for i in names:
			dic[i]=[]

		n_dic = {}
		
		for i,j in zip(names,keys):
			n_dic[i]=j
		top_label = str(n_dic)[1:-1]
		top_label=top_label.replace("'","")

		split = vid.split('.')
		vid_name = os.path.basename(split[0])
		cap = cv2.VideoCapture(vid)
		fps =  cap.get(cv2.CAP_PROP_FPS)
		length = int((int(time)*fps*60)+15)

		def onChange(trackbarValue):
			cap.set(cv2.CAP_PROP_POS_FRAMES,trackbarValue)
			err,img = cap.read()
			h,w,d = img.shape
			
			if cnt == 1:
				n_img = cv2.putText(img, name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX , 0.7,(255, 255, 255), 2, cv2.LINE_AA, False)
				cv2.imshow(top_label, n_img)

			if cnt == 0:
				cv2.imshow(top_label, img)

			if reset == 1:
				n_img = cv2.putText(img, 'reset', (w-70, 20), cv2.FONT_HERSHEY_SIMPLEX , 0.7,  
		                 (255, 255, 255), 2, cv2.LINE_AA, False)
				cv2.imshow(top_label, n_img)

			if show == 1:
				dic_temp = {}
				temp = list(dic.values())
				for i,j in zip(range(len(temp)),dic.keys()):
					l = temp[i][-2:]
					l.append(str(len(temp[i])))
					dic_temp[j]=l
				n_img = cv2.putText(img, str(dic_temp), (int(w/2)-len(str(dic_temp)),20), cv2.FONT_HERSHEY_SIMPLEX , 0.7,  
		                 (255, 255, 255), 2, cv2.LINE_AA, False)
				cv2.imshow(top_label, n_img)


		def getmax(data):
			m_list = []
			t = list(data.values())
			for i in t:
				if i != []:
					m_list.append(max(i))
				else:
					m_list.append(0)
			return m_list.index(max(m_list))

		global cnt
		global reset
		global l
		global name
		global show

		cnt = 0
		reset = 0
		show = 0
		compare = []
		cv2.namedWindow(top_label, cv2.WINDOW_NORMAL)
		cv2.createTrackbar( 'pos', top_label, 0, length, onChange )
		onChange(0)

		while cap.isOpened():

			k = cv2.waitKey(0) #& 0xff

			for i in range(len(keys)):
				if k == ord(keys[i]):
					if compare == [] or keys[i] == compare[0]:
						compare.append(keys[i])
						pos = cv2.getTrackbarPos('pos',top_label)
						dic[list(dic)[i]].append(pos)
						name = list(dic)[i]
						l = len(list(dic.values())[i])

						if l % 2 != 0:
							cnt = 1

						if l % 2 == 0:
							cnt = 0
							compare = []

						onChange(pos)

			if k == 8:
				reset = 1
				pos = cv2.getTrackbarPos('pos',top_label)
				onChange(pos)
				if compare != []:
					compare = compare[:-1]
				i = getmax(dic)
				del dic[names[i]][-1]
				
				if cnt == 1:
					cnt = 0
				else:
					cnt = 1

				reset = 0

			if k == ord('p'):
				show = 1
				pos = cv2.getTrackbarPos('pos',top_label)
				onChange(pos)
				show = 0

			elif k == 27:
				 
				f = {'experiment':os.path.basename(os.path.dirname(videos[0])),'video':vid_name}
				for i,j in zip(dic.keys(),dic.values()):
					f[i] = j
				frames.append(f)

				temp_list = list(dic.values())
				d_list = []
				for d in temp_list:
					d_list.append(0)

				minute = 0

				for i in range(int(fps*60),int(int(time)*fps*60)+int(fps*60),int(fps*60)):
					minute += 1
					tot_list = []
					d = {'experiment':os.path.basename(os.path.dirname(videos[0])),'video':vid_name,'minute':minute}

					for val in range(len(temp_list)):
						tot,diff = write(temp_list[val],i)
						tot = tot+d_list[val]
						d_list[val] = diff
						tot_list.append(tot)

					for key,val in zip(dic.keys(),tot_list):	
						d[key]=val
					
					data.append(d)

				print('processed video:',vid_name)
				break

		cap.release()
		cv2.destroyAllWindows()

	# write a csv with all values
	exp_name = os.path.basename(os.path.dirname(videos[0]))
	df = pd.DataFrame.from_dict(data)
	filename = os.path.join(project_path, 'Hand_label_'+ exp_name + '.csv')
	print(filename)
	if os.path.isfile(filename):
		cnt = 1
		while True:
			new_filename = os.path.join(project_path, 'Hand_label_'+ exp_name + '_' + str(cnt) + '.csv')
			cnt += 1
			if os.path.isfile(new_filename):
				continue
			else:
				filename = new_filename
				break
	df.to_csv(filename, index=False)

	# write a csv with all values
	df = pd.DataFrame.from_dict(frames)
	filename2 = os.path.join(project_path, 'Frames_'+ exp_name + '.csv')
	if os.path.isfile(filename2):
		cnt = 1
		while True:
			new_filename2 = os.path.join(project_path, 'Frames_'+ exp_name + '_' + str(cnt) + '.csv')
			cnt += 1
			if os.path.isfile(new_filename2):
				continue
			else:
				filename2 = new_filename2
				break
	df.to_csv(filename2, index=False)

# project_path = '/Volumes/WD_My_Passport/Masterarbeit/NOR'
# videos = ['/Volumes/WD_My_Passport/Masterarbeit/NOR/B3_ORT_testing_070421/11.MP4']
# time = 2
# names = ['o1','o2']
# keys = ['d','f']
# hand_label(project_path,videos,time,names,keys)
