import cv2
import numpy as np
import os
import math

THRESHOLD = 0
DIRNAME = os.path.dirname(__file__)
DIR_YOSEMITE = os.path.join(DIRNAME, 'yosemite')
yosemite1 = os.path.join(DIR_YOSEMITE, 'Yosemite1.jpg')
yosemite2 = os.path.join(DIR_YOSEMITE, 'Yosemite2.jpg')
img1_orig = cv2.imread(yosemite1)
img2_orig = cv2.imread(yosemite2)
img1 = cv2.imread(yosemite1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(yosemite2, cv2.IMREAD_GRAYSCALE)

Ix = cv2.Sobel(img1, cv2.CV_32F, 1, 0, ksize=3)
Iy = cv2.Sobel(img1, cv2.CV_32F, 0, 1, ksize=3)
Ix2 = np.matrix(Ix * Ix, dtype=np.float32)
Iy2 = np.matrix(Iy * Iy, dtype=np.float32)
IxIy = np.matrix(Ix * Iy, dtype=np.float32)

def detectFeatures(img):    
    grayscale = cv2.cvtColor(img1_orig,cv2.COLOR_BGR2GRAY)
    keypoints = grayscale.copy()
    keypoints[:] = 0
    for i in range(grayscale.shape[0]):
        for j in range(grayscale.shape[1]):
            det = Ix2[i, j] * Iy2[i, j] - (IxIy[i, j])**2
            trace = Ix2[i, j] + Iy2[i, j]
            R = det / trace
            if not math.isnan(R) and R > THRESHOLD:
                keypoints[i, j] = grayscale[i, j]
    local_max = get_local_max(keypoints)
    return local_max

def get_local_max(img):
    local_max = img.copy()
    local_max[:] = 0
    for i in range(img.shape[0]):        
        for j in range(img.shape[1]):
            up_left = img[i-1, j-1] if (i > 0 and j > 0) else 0
            up = img[i-1, j] if i > 0 else 0
            up_right = img[i-1, j+1] if (i > 0 and j < img.shape[1]-1) else 0
            left = img[i, j-1] if j > 0 else 0
            centre = img[i, j]
            right = img[i, j+1] if j < img.shape[1]-1 else 0
            bottom_left = img[i+1, j-1] if (i < img.shape[0]-1 and j > 0) else 0
            bottom = img[i+1, j] if i < img.shape[0]-1 else 0
            bottom_right = img[i+1, j+1] if (i < img.shape[0]-1 and j < img.shape[1]-1) else 0
            padded_img = np.matrix([[up_left, up, up_right], [left, centre, right], [bottom_left, bottom, bottom_right]])
            local_max[i,j] = img[i,j] if img[i,j] == padded_img.max() else 0
    return local_max

features = detectFeatures(img1_orig.copy())
cv2.imshow('features', features)
cv2.waitKey(0)
cv2.destroyAllWindows()
