########################################################################
#3. SCRIPT FOR OBTAINING SPECIMEN STRAIN FROM IMPACT VIDEO
########################################################################

import sys
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl 
from start_frame import start_frame, vidName

# Read video file from script start_frame.py
video = cv2.VideoCapture(vidName)  # Read video file
out = cv2.VideoWriter('Tracking_Example.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (840,312)) # Setup video output

# Isolate first frame of video and select regions of interest
first_frame = cv2.imread('frame1.jpg')        # query image
first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY) # convert to grayscale
ROIs = cv2.selectROIs("Select regions to be tracked",first_frame,False)

x1,y1,width1,height1 = ROIs[0]
x2,y2,width2,height2 = ROIs[1]

bar1_img = first_frame[y1: y1+height1, x1: x1+width1] # Crop first frame to show only image of bar 1 from selected region
bar2_img = first_frame[y2: y2+height2, x2: x2+width2] # Crop first frame to show only image of bar 2 from selected region

# Identify features within chosen regions of interest
sift = cv2.SIFT_create()
kp_image1, desc_image1 = sift.detectAndCompute(bar1_img, None)  # Find key points and descriptors of bar 1
kp_image2, desc_image2 = sift.detectAndCompute(bar2_img, None)  # Find key points and descriptors of bar 2

# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# Initialise looping counters
current_frame = 0
x_bar1 = []
x_bar2 = []
strainTime = []
current_time = 0

roi_kp = cv2.drawKeypoints(bar1_img,kp_image1,bar1_img)
cv2.imshow("keypoints", roi_kp)

while True:
    ret,frame = video.read()
    
    if ret:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # train image (image to be compared with query)

        current_frame += 1 # For each new frame, increase frame count by 1
  
        if current_frame >= start_frame:    # Once frame count has reached start of impact, start tracking the regions of interest
            
            kp_grayframe, desc_grayframe = sift.detectAndCompute(gray_frame, None)
            matches1 = flann.knnMatch(desc_image1, desc_grayframe, k=2)
            matches2 = flann.knnMatch(desc_image2, desc_grayframe, k=2)

            good_points1 = []
            for m, n in matches1:  # m: query image, n: object from trained image
                if m.distance < 0.8*n.distance:
                    good_points1.append(m)

            good_points2 = []
            for m, n in matches2:  # m: query image, n: object from trained image
                if m.distance < 0.8*n.distance:
                    good_points2.append(m)
            
            matches_demo = cv2.drawMatches(bar1_img, kp_image1, gray_frame, kp_grayframe, good_points1, gray_frame)

            # Homography
            if len(good_points1) and len(good_points2) > 8: # If number of good matches is above a chosen threshold, track region
                ### REGION 1###
                query_pts1 = np.float32([kp_image1[m.queryIdx].pt for m in good_points1]).reshape(-1,1,2)  # extracting position of good points of query image
                train_pts1 = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points1]).reshape(-1,1,2)

                matrix1, mask1 = cv2.findHomography(query_pts1, train_pts1, cv2.RANSAC, 5.0)
                matches_mask1 = mask1.ravel().tolist()
                
                # Perspective tansform
                h1, w1 = bar1_img.shape
                pts1 = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2)
                dst1 = cv2.perspectiveTransform(pts1, matrix1)

                homography1 = cv2.polylines(frame, [np.int32(dst1)], True, (0,255,0), 2)
                ##########################################################################################
                ### REGION 2 ###
                query_pts2 = np.float32([kp_image2[m.queryIdx].pt for m in good_points2]).reshape(-1,1,2)  # extracting position of good points of query image
                train_pts2 = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points2]).reshape(-1,1,2)

                matrix2, mask2 = cv2.findHomography(query_pts2, train_pts2, cv2.RANSAC, 5.0)
                matches_mask2 = mask2.ravel().tolist()
                
                # Perspective tansform
                h2, w2 = bar2_img.shape
                pts2 = np.float32([[0,0], [0,h2], [w2,h2], [w2,0]]).reshape(-1,1,2)
                dst2 = cv2.perspectiveTransform(pts2, matrix2)

                homography2 = cv2.polylines(frame, [np.int32(dst2)], True, (0,255,0), 2)
                ###########################################################################################
                x1_bar1 = dst1[2]
                x1_bar1 = x1_bar1[0]
                x1_bar1 = x1_bar1[0]
                x_bar1.append(x1_bar1)

                x1_bar2 = dst2[0]
                x1_bar2 = x1_bar2[0]
                x1_bar2 = x1_bar2[0]
                x_bar2.append(x1_bar2)
                
                strainTime.append(current_time)
                current_time += 6.25e-6
                
                cv2.imshow("Homography", homography2)
                out.write(homography2)

                key = cv2.waitKey(-1)
                if key and 0xFF == 27:
                    break
            else:
                cv2.imshow("Homography", gray_frame)
                key = cv2.waitKey(-1)
                if key and 0xFF == 27:
                    break
            
            cv2.imshow("Matches", matches_demo)
            
            key = cv2.waitKey(-1)
            if key and 0xFF == 27:
                break
    else:
        break

# Calculate specimen strain using coordinates of regions of interest
spec_lengthOriginal = x_bar2[0] - x_bar1[0]
x_bar1 = np.array(x_bar1)
x_bar2 = np.array(x_bar2)
spec_lengthCurrent = x_bar2 - x_bar1

engStrain = (spec_lengthOriginal-spec_lengthCurrent)/spec_lengthOriginal
strainTime = np.array(strainTime) * 1e6

# Plot result
fig2=plt.figure(facecolor='white', figsize=(10,8))
plt.plot(strainTime,engStrain,'b--',label='Specimen engineering strain',markersize=7, markerfacecolor='w',markeredgecolor='r',markeredgewidth=2)
plt.xlim(0, np.amax(strainTime))
plt.legend(loc='best',numpoints=1)
plt.xlabel(r'$Time\ [microseconds]$',fontsize=24)
plt.ylabel(r'$Engineering\ strain\ [-]$',fontsize=24)
plt.grid( b=None,which='major', axis='both',linewidth=1)
plt.grid( b=None,which='minor', axis='both',linewidth=0.2)
plt.show()

video.release()
cv2.destroyAllWindows()