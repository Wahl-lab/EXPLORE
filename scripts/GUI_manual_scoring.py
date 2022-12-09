#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------
# ORT analysis - script GUI manual scoring - developed by Victor Ibañez
# 03.04.2021
# -------------------------------------------------------------------------------------


# -----------------------------------------------------
# import libraries
# -----------------------------------------------------

from tkinter import *
import tkinter.filedialog as fd
import os

# -----------------------------------------------------
# create GUI 
# -----------------------------------------------------

def GUI_manual_scoring():
	
	# get entry values
	def get_values():
		return [entry.get() for entry in entries]
	
	# create main interface
	def call_dic(dic):
		
		# grab dictionary structure 
		for key, value in dic.items():
			
			for i in range(len(value)):
				
				# create labels
				if value[i][0] == 'Label':
					lbl=Label(window, text=value[i][1][1])
					lbl.place(x=int(win_w*value[i][1][0]), y=int(win_h*key))
					lbl.config(font=("TkDefaultFont", f_size, "italic"))
				
				# create entries
				if value[i][0] == 'Entry':
					txtfld=Entry(window, bd=2, width=value[i][1][1])
					txtfld.place(x=int(win_w*value[i][1][0]), y=int(win_h*key))
					entries.append(txtfld)
				
				# create buttons
				if value[i][0] == 'Button':
					if value[i][1][1] == 'path':
						btn=Button(window, text="search", command=open_path)
					if value[i][1][1] == 'vid':
						btn=Button(window, text="search", command=open_vid)
					if value[i][1][1] == 'submit':
						btn=Button(window, text="start", command=submit)

					btn.place(x=int(win_w*value[i][1][0]), y=int(win_h*key))
					btn.config(font=("TkDefaultFont", f_size, "italic"))
	
	# function to get project path
	def open_path():
		global project_path

		project_path = fd.askdirectory()

		lbl=Label(window, text=project_path.ljust(1000))
		lbl.place(x=int(win_w*0.05), y=int(win_h*0.26))
		lbl.config(font=("Arial", f_size-2))
	
	# function to get video paths
	def open_vid():
		global videos

		videos = fd.askopenfilenames()
		
		# take first video and get basename, then calculate the nb. of video name strings fits into one horizontal line of the GUI
		name0 = os.path.basename(videos[0]).split('.')[0]
		l = len(name0)
		nb_items = int((win_w/(l+2))/(f_size/1.5))
		
		# add label below button in several rows if necessary
		for i in range(len(videos)):
			name = os.path.basename(videos[i]).split('.')[0]

			if i < nb_items:
				lbl1=Label(window, text=name.ljust(1000))
				lbl1.place(x=int(win_w*0.05)+(i*int(win_w*0.05)), y=int(win_h*0.41))
				lbl1.config(font=("Arial", f_size-2))
				lbl2=Label(window, text=''.ljust(1000))
				lbl2.place(x=int(win_w*0.05), y=int(win_h*0.44))
				lbl2.config(font=("Arial", f_size-2))
			if i >= nb_items:
				lbl3=Label(window, text=name.ljust(1000))
				lbl3.place(x=int(win_w*0.05)+((i-nb_items)*int(win_w*0.05)), y=int(win_h*0.44))
				lbl3.config(font=("Arial", f_size-2))

	# start analysis
	def submit():
		global project_name
		global video_length
		global objects
		global o_keys

		entry_list = get_values()
		project_name = entry_list[0]
		video_length = entry_list[1]
		objects = entry_list[2:13:2]
		if '' in objects:
			objects.remove('')
		o_keys = entry_list[3:14:2]
		if '' in o_keys:
			o_keys.remove('')

		window.destroy()

	# ------------------------------------------------------------------------------------------------------
	# 				*** MAIN SETTINGS ***
	# ------------------------------------------------------------------------------------------------------
	
	window=Tk()
	
	# get screen dimensions info
	width = window.winfo_screenwidth()
	height = window.winfo_screenheight()
	win_w = int(width/2)
	win_h = int(height/1.15)
	
	# default font size compared to screen dimensions
	norm_w = 1280
	norm_f_size = 14
	
	# calculate fontsize depending on different screen dimensions than default 
	f_size = round((norm_w*norm_f_size)/width)
	
	# main structure of GUI
	main = {0.03:[['Label',[0.05,'project name:']],['Label',[0.55,'cutting length videos (min):']]],
			0.08:[['Entry',[0.05,30]],['Entry',[0.55,5]]],
			0.16:[['Label',[0.05,'choose project path:']]],
			0.21:[['Button',[0.05,'path']]],
			0.31:[['Label',[0.05,'choose videos for manual scoring:']]],
			0.36:[['Button',[0.05,'vid']]],
			0.57:[['Label',[0.05,'add objects:']],['Label',[0.4,'Add key (a-z; "p" reserved):']]],
			0.62:[['Label',[0.05,'1.     name:']],['Entry',[0.2,10]],['Label',[0.4,'key:']],['Entry',[0.48,3]]],
			0.67:[['Label',[0.05,'2.     name:']],['Entry',[0.2,10]],['Label',[0.4,'key:']],['Entry',[0.48,3]]],
			0.72:[['Label',[0.05,'3.     name:']],['Entry',[0.2,10]],['Label',[0.4,'key:']],['Entry',[0.48,3]]],
			0.77:[['Label',[0.05,'4.     name:']],['Entry',[0.2,10]],['Label',[0.4,'key:']],['Entry',[0.48,3]]],
			0.82:[['Label',[0.05,'5.     name:']],['Entry',[0.2,10]],['Label',[0.4,'key:']],['Entry',[0.48,3]]],
			0.87:[['Label',[0.05,'6.     name:']],['Entry',[0.2,10]],['Label',[0.4,'key:']],['Entry',[0.48,3]]],
			0.94:[['Button',[0.05,'submit']]]}
				
	# create list to grab entries of main 
	entries = []
	
	# call main function
	call_dic(main)

	window.title('Create a new Project!')
	window.geometry(str(win_w)+'x'+str(win_h)+'+10+10')
	window.mainloop()


	return project_path, project_name, video_length, videos, objects, o_keys

