import sys
from tkinter import *

ap = Tk()

ap.geometry("600x400+200+200")
ap.title("tkke")
while True:
    ap.update()
    a = ap.winfo_geometry()
    print(ap.winfo_height())
#print(ap.winfo_width())
    print(ap.winfo_x)
    print("this is {}".format(a))
    print(ap.winfo_geometry())
#print(ap.winfo_reheight())
#print(ap.winfo_rewidth())


ap.mainloop()
