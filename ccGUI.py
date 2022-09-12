from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import ccMain
import ccDataHandler
import cv2
import atexit
import numpy as np

#define custom typesets
LARGE_FONT = ('Verdana 15 underline')
BOLD_FONT = ('Verdana 9 bold')
global userFlag
userFlag = 0

global iconPA
iconPA = []

if __name__ == '__main__': #code only executed to run as a program not when simply imported as a module
	root = Tk()
	root.geometry('375x375')
	root.wm_title("Food Identifier")
	root.resizable(width=False, height=False)

	def exit_handler():
		os.remove('userIcon.jpg') #remove icon image at exit

	def open():
		global userFlag #use global value of userFlag
		global iconPA

		#open file explorer for tkinter
		img = filedialog.askopenfilename(initialdir = os.path.dirname(os.path.abspath(__file__)),title = "Select an image to upload to model",filetypes = (("jpeg files","*.jpg *.jpeg"),))
		
		#use PIL to convert image to usable format
		selection = ImageTk.PhotoImage(Image.open(img).resize((220, 220), Image.ANTIALIAS))
		imgLabel.config(image=selection)
		imgLabel.photo_ref=selection #keep reference to new image

		imgIcon = Image.open(img).resize((32, 32), Image.ANTIALIAS) #icon image for passing into ML algorithm
		imgIcon.save('userIcon.jpg','JPEG')

		iconPA = cv2.imread('userIcon.jpg')
		iconPA = cv2.cvtColor(iconPA, cv2.COLOR_BGR2RGB) #cv2.imread converts image to BGR, this converts it back to RGB
		
		iconPA = np.concatenate(iconPA)

		tempArray = []
		for i in range(3):
			for j in range(len(iconPA)):
				tempArray.append(iconPA[j][i])

		iconPA = tempArray
		userFlag = 1

	upload = Image.open('placeholderImg.png').resize((220, 220), Image.ANTIALIAS) #scale image, keep ratio
	render = ImageTk.PhotoImage(upload)

	titleLabel = Label(root, text="Food Identifier\n", font=LARGE_FONT)
	imgLabel = Label(root, image=render)
	uploadButton = Button(root, text="Upload image..", command=open)
	spacerLabel = Label(root, text='')
	runButton = Button(root, text="Run model", activeforeground="green", fg="green", font=BOLD_FONT, command=lambda: ccMain.startModel(userFlag, iconPA))

	titleLabel.pack()
	uploadButton.pack()
	imgLabel.pack()
	spacerLabel.pack()
	runButton.pack()

	atexit.register(exit_handler) #define functionality at program exit

	root.mainloop()
