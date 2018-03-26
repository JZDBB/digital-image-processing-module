#!/usr/bin/python3
# 2017.11.11 01:44:37 CST
# 2017.11.12 00:09:14 CST
"""
使用Sift特征点检测和匹配查找场景中特定物体。
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
MIN_MATCH_COUNT = 15

def is_in_box(axis, box):
    flag = False
    x0, y0 = box[0][0]
    x1, y1 = box[2][0]

    if axis.pt[0] > x0 and axis.pt[0] < x1:
        if axis.pt[1] > y0 and axis.pt[1] < y1:
            flag = True
    return flag

imgname1 = "template1.jpg"
imgname2 = "test.jpg"

## (1) prepare data
img1 = cv2.imread(imgname1)
img2 = cv2.imread(imgname2)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


## (2) Create SIFT object
sift = cv2.xfeatures2d.SIFT_create()

## (3) Create flann matcher
matcher = cv2.FlannBasedMatcher(dict(algorithm = 1, trees = 5), {})

## (4) Detect keypoints and compute keypointer descriptors
kpts1, descs1 = sift.detectAndCompute(gray1,None)
kpts2, descs2 = sift.detectAndCompute(gray2,None)

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
    h,w = img1.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    ## 绘制边框
    cv2.polylines(canvas,[np.int32(dst)],True,(0,255,0),3, cv2.LINE_AA)
    # i = 0
    # while kpts2[i]:
    #     if is_in_box(kpts2[i], dst):
    #         del kpts2[i]
    #         descs2 = np.delete(descs2, i, 0)
    #         continue
    #     i = i + 1

    a = np.array([[[dst[0][0][0], dst[0][0][1]], [dst[2][0][0] + w, dst[0][0][1]], [dst[2][0][0] + w, dst[2][0][1] + h], [dst[0][0][0], dst[2][0][1] + h]]], dtype=np.int32)
    cv2.fillPoly(gray2, a, 255)
    plt.imshow(gray2, cmap='gray')
    plt.show()
    # kpts2, descs2 = sift.detectAndCompute(gray2, None)
    # matches = matcher.knnMatch(descs1, descs2, 2)
    # # Sort by their distance.
    # matches = sorted(matches, key=lambda x: x[0].distance)
    # ## (6) Ratio test, to get good matches.
    # good = [m1 for (m1, m2) in matches if m1.distance < 0.7 * m2.distance]


## (8) drawMatches
matched = cv2.drawMatches(img1,kpts1,canvas,kpts2,good,None)#,**draw_params)

## (9) Crop the matched region from scene
h,w = img1.shape[:2]
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)
perspectiveM = cv2.getPerspectiveTransform(np.float32(dst),pts)
found = cv2.warpPerspective(img2,perspectiveM,(w,h))

## (10) save and display
cv2.imwrite("matched.png", matched)
cv2.imwrite("found.png", found)
cv2.imshow("matched", matched)
cv2.imshow("found", found)
cv2.waitKey();cv2.destroyAllWindows()