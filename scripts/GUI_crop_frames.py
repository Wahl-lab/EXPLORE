import cv2
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image

# get a frame of video
def grep_frame(video):
	vidcap = cv2.VideoCapture(video)
	success,image = vidcap.read()
	count = 1

	while success:
		if count == 50:
			return image
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
		cv2.imshow("draw rectangle around surface / 'r' for again / 'c' for crop", image)

def draw_grid(image, h, w):
	global v1, v2, v3, v4, root
	
	b,g,r = cv2.split(image)
	img = cv2.merge((r,g,b))

	root = Tk()

	im = Image.fromarray(img)
	imgtk = ImageTk.PhotoImage(image=im)

	canvas = Canvas(root, width=w, height=h)
	canvas.pack()
	canvas.create_image(w, h, image=imgtk, anchor='se')

	quit_button = Button(root, text = "Save", command = submit, anchor = 'w',
	                    width = 4)#, activebackground = "#33B5E5")
	quit_button_window = canvas.create_window(10, 10, anchor='nw', window=quit_button) 

	s1,s2,s3,s4 = 0,0,0,0
	
	# get values
	v1 = IntVar()
	v2 = IntVar()
	v3 = IntVar()
	v4 = IntVar()
	C1 = Checkbutton(root, text = "", variable = v1)
	C2 = Checkbutton(root, text = "", variable = v2) 
	C3 = Checkbutton(root, text = "", variable = v3)
	C4 = Checkbutton(root, text = "", variable = v4)  
	
	# 
	C1.place(x=int(w/5), y=int(h/4))
	C2.place(x=int(w/5), y=int(h/4)*3)
	C3.place(x=int(w/5)*3, y=int(h/4))
	C4.place(x=int(w/5)*3, y=int(h/4)*3)
	root.title('Select your objects')
	root.mainloop()

def submit():
	global s1, s2, s3, s4
	#project_name = txtfld.get()
	s1 = v1.get()
	s2 = v2.get()
	s3 = v3.get()
	s4 = v4.get()
	root.destroy()

def main(video):
	global image
	global ref_point, crop
	
	image = grep_frame(video)   		
	ref_point = [] 
	crop = False  
	# load the image, clone it, and setup the mouse callback function 
	 
	clone = image.copy() 
	cv2.namedWindow("draw rectangle around surface / 'r' for again / 'c' for crop", cv2.WINDOW_NORMAL)
	cv2.setMouseCallback("draw rectangle around surface / 'r' for again / 'c' for crop", shape_selection) 
	cv2.startWindowThread()
	  
	# keep looping until the 'q' key is pressed 
	while True: 
	    # display the image and wait for a keypress 
		cv2.imshow("draw rectangle around surface / 'r' for again / 'c' for crop", image) 
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

	return ref_point # s1, s2, s3, s4, 
