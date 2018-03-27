#!/usr/bin/python3
# 2017.11.11 01:44:37 CST
# 2017.11.12 00:09:14 CST
"""
使用Sift特征点检测和匹配查找场景中特定物体。
"""

import cv2
from numpy import *
import numpy as np
from matplotlib import pyplot as plt
MIN_MATCH_COUNT = 4

# def shape_affine(object, ):




def is_in_box(axis, box):
    flag = False
    x0, y0 = box[0][0]
    x1, y1 = box[2][0]

    if axis.pt[0] > x0 and axis.pt[0] < x1:
        if axis.pt[1] > y0 and axis.pt[1] < y1:
            flag = True
    return flag

imgname1 = "temple_last.jpg"
imgname2 = "606.jpg"
imgname3 = "template1_1.jpg"

## (1) prepare data
img1 = cv2.imread(imgname1)
img2 = cv2.imread(imgname2)
img3 = cv2.imread(imgname3)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)


## (2) Create SIFT object
sift = cv2.xfeatures2d.SIFT_create()

## (3) Create flann matcher
matcher = cv2.FlannBasedMatcher(dict(algorithm = 1, trees = 5), {})

## (4) Detect keypoints and compute keypointer descriptors
kpts1, descs1 = sift.detectAndCompute(gray1,None)
kpts2, descs2 = sift.detectAndCompute(gray2,None)
kpts3, descs3 = sift.detectAndCompute(gray3,None)

## (5) knnMatch to get Top2
matches = matcher.knnMatch(descs1, descs2, 2)
# Sort by their distance.
matches = sorted(matches, key = lambda x:x[0].distance)

## (6) Ratio test, to get good matches.
good = [m1 for (m1, m2) in matches if m1.distance < 0.7 * m2.distance]
print(len(good))

canvas = img2.copy()

## (7) find homography matrix
## 当有足够的健壮匹配点对（至少4个）时
if len(good)>MIN_MATCH_COUNT:
    ## 从匹配中提取出对应点对
    ## (queryIndex for the small object, trainIndex for the scene )
    src_pts = np.float32([ kpts1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kpts2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    ## find homography matrix in cv2.RANSAC using good match points
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    ## 掩模，用作绘制计算单应性矩阵时用到的点对
    #matchesMask2 = mask.ravel().tolist()
    ## 计算图1的畸变，也就是在图2中的对应的位置。
    h, w = img1.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    print(dst)
    # found = gray2[np.int(np.min(dst[:, :, 1])):np.int(np.max(dst[:, :, 1])), np.int(np.min(dst[:, :, 0])):np.int(np.max(dst[:, :, 0]))]
    # plt.imshow(found, cmap='gray')
    # plt.show()
    A = mat([[393, 212, 1],
             [393, 313, 1],
             [784, 313, 1]])
    B = mat([[np.float(dst[0][0][0]), np.float(dst[0][0][1])],
             [np.float(dst[1][0][0]), np.float(dst[1][0][1])],
             [np.float(dst[2][0][0]), np.float(dst[2][0][1])]])
    X = (A.I)*B
    A1 = mat([[0, 0, 1],
              [0, 884, 1],
              [1168, 884, 1],
              [1168, 0, 1]])
    result = A1*X

    pt1 = list([[result[0][:,0].max(), result[0][:,1].max()]])
    pt2 = list([[result[1][:,0].max(), result[1][:,1].max()]])
    pt3 = list([[result[2][:,0].max(), result[2][:,1].max()]])
    pt4 = list([[result[3][:,0].max(), result[3][:,1].max()]])
    dst_result = array([pt1, pt2, pt3, pt4])
    print(dst_result)

    cv2.polylines(canvas,[np.int32(dst_result)],True,(0,255,0),3, cv2.LINE_AA)


    ## (8) drawMatches
    matched = cv2.drawMatches(img1,kpts1,canvas,kpts2,good,None)#,**draw_params)

    ## (9) Crop the matched region from scene
    # h,w = img1.shape[:2]
    # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    # dst = cv2.perspectiveTransform(pts,M)
    perspectiveM = cv2.getPerspectiveTransform(np.float32(dst),pts)
    found = cv2.warpPerspective(img2,perspectiveM,(w,h))

    ## (10) save and display
    cv2.imwrite("matched.png", canvas)
    cv2.imwrite("found.png", found)
    cv2.imshow("matched", matched)
    cv2.imshow("found", found)
    cv2.waitKey();cv2.destroyAllWindows()