import cv2
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image


def grep_frame(video, ref):
	vidcap = cv2.VideoCapture(video)
	success,image = vidcap.read()
	count = 1

	while success:
		if count == 50:
			res1 = image[ref[0][1]:ref[1][1], ref[0][0]:ref[1][0]]
			return res1
			break

		success,image = vidcap.read()
		count += 1



def shape_selection(event, x, y, flags, param): 
	global ref_point
    # grab references to the global variables  
  
    # if the left mouse button was clicked, record the starting 
    # (x, y) coordinates and indicate that cropping is being performed 
	if event == cv2.EVENT_LBUTTONDOWN: 
		ref_point = [(x, y)] 
  
    # check to see if the left mouse button was released 
	elif event == cv2.EVENT_LBUTTONUP: 
        # record the ending (x, y) coordinates and indicate that 
        # the cropping operation is finished 
		ref_point.append((x, y)) 
  
        # draw a rectangle around the region of interest 
		cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2) 
		cv2.imshow("draw rectangle around current object / 'r' for again / 'c' for crop", image)
		#cv2.putText(image, o, (10, 30), cv2.FONT_HERSHEY_SIMPLEX , 1,(255, 255, 255), 3, cv2.LINE_AA, False)

def main(video, ref, o):

	global image
	global ref_point, crop
	image = grep_frame(video,ref)   		
	ref_point = [] 

	crop = False  
	# load the image, clone it, and setup the mouse callback function 
	 
	clone = image.copy() 
	cv2.namedWindow("draw rectangle around current object / 'r' for again / 'c' for crop",cv2.WINDOW_NORMAL) 
	cv2.setMouseCallback("draw rectangle around current object / 'r' for again / 'c' for crop", shape_selection) 
	cv2.startWindowThread()
	  
	# keep looping until the 'q' key is pressed 
	while True: 
	    # display the image and wait for a keypress 
		cv2.imshow("draw rectangle around current object / 'r' for again / 'c' for crop", image) 
		cv2.putText(image, o, (10, 30), cv2.FONT_HERSHEY_SIMPLEX , 1,(255, 255, 255), 3, cv2.LINE_AA, False)
		key = cv2.waitKey(1) & 0xFF
	  
	    # press 'r' to reset the window 
		if key == ord("r"): 
			image = clone.copy() 
	  
	    # if the 'c' key is pressed, break from the loop 
		elif key == ord("c"): 
			break

	if len(ref_point) == 2: 

		crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
		
		h,w,d = crop_img.shape
		h_x1, h_x2, h_y1, h_y2 = 0, w, int(h/2), int(h/2)
		v_x1, v_x2, v_y1, v_y2 = int(w/2), int(w/2), h, 0

		#draw_grid(crop_img, h, w)

		y1 = ref_point[0][1] 
		y2 = ref_point[1][1] 
		x1 = ref_point[0][0] 
		x2 = ref_point[1][0]
		
	# close all open windows 
	cv2.destroyAllWindows()

	return ref_point
