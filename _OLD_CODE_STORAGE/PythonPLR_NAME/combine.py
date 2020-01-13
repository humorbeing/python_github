import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk, ImageGrab


#Set up GUI
window = tk.Tk()  #Makes main window
window.wm_title("Digital Microscope")
window.config(background="#FFFFFF")

#Graphics window
imageFrame = tk.Frame(window, width=600, height=500)
imageFrame.grid(row=0, column=0, padx=10, pady=2)

#Capture video frames
lmain = tk.Label(imageFrame)
lmain.grid(row=0, column=0)



def show_frame():
    while True:
        window.update()
        x = window.winfo_x() - 500
        y = window.winfo_y()

        imggrab = ImageGrab.grab(bbox=(x,y,500,500)) #bbox specifies specific region (bbox= x,y,width,height)
        toimg = np.array(imggrab)
        frame = cv2.cvtColor(toimg, 2)
        #frame = cv2.flip(frame, 1)
        #cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(5, show_frame)
    print('is it looping?')



#Slider window (slider controls stage position)
sliderFrame = tk.Frame(window, width=600, height=100)
sliderFrame.grid(row = 600, column=0, padx=10, pady=2)


show_frame()  #Display 2
window.mainloop()  #Starts GUI
