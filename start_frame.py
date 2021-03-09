########################################################################
#2. SCRIPT FOR SELECTING IMPACT VIDEO AND CHOOSING STARTING FRAME
########################################################################
import cv2
import numpy as np 

import tkinter as tk
from tkinter import filedialog
import os

# Get video name
root = tk.Tk()
root.withdraw()
print("Select a video file")
vidName = filedialog.askopenfilename()
vidChosen = ('%s' % os.path.basename(vidName))
print("Video chosen: ", vidChosen)
root.destroy()

# Open video file
cap = cv2.VideoCapture(vidName)

# Define starting frame of impact
start_frame = 0 
while True:
    ret, frame = cap.read()
    
    if ret:
        start_frame += 1
        cv2.imshow("Go to starting frame of impact and hit esc.", frame)

        key = cv2.waitKey(0)
        if key == 27:
            cv2.imwrite('frame1.jpg', frame)  # Save image of first frame of impact
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
    