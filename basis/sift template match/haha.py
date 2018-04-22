import cv2
import numpy as np
MIN_MATCH_COUNT = 4

imgname1 = "test1.jpg"

## (1) prepare data
img1 = cv2.imread(imgname1)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

## (2) Create SIFT object
sift = cv2.xfeatures2d.SIFT_create()

## (4) Detect keypoints and compute keypointer descriptors
kpts1, descs1 = sift.detectAndCompute(gray1,None)

times = 0
for i in range(len(kpts1)):
    if i % 3 == 0:
        del kpts1[i - times]
        times = times + 1
times = 0
for i in range(len(kpts1)):
    if i % 3 == 0:
        del kpts1[i - times]
        times = times + 1

## (8) drawMatches
# matched = cv2.drawKeypoints(img1,kpts1)
# matched = cv2.drawKeypoints(img1,kpts1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
matched = cv2.drawKeypoints(img1, kpts1, img1, (0, 0 ,0))

## (10) save and display
cv2.imwrite("matched.png", matched)
