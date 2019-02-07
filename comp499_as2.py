import cv2
import numpy as np
import os
import math

DIRNAME = os.path.dirname(__file__)
DIR_YOSEMITE = os.path.join(DIRNAME, 'yosemite')
yosemite1 = os.path.join(DIR_YOSEMITE, 'Yosemite1.jpg')
yosemite2 = os.path.join(DIR_YOSEMITE, 'Yosemite2.jpg')
img1_orig = cv2.imread(yosemite1)
img2_orig = cv2.imread(yosemite2)
img1 = cv2.imread(yosemite1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(yosemite2, cv2.IMREAD_GRAYSCALE)

# gray = cv2.cvtColor(img1_orig,cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)
# dst = cv2.cornerHarris(gray, 2, 3, 0.04)
# print('dst.max(): ' + str(dst.max()))
# print(0.01 * dst.max())
# img1_orig[dst > 0.01 * dst.max()] = [0,0,255]
# cv2.imshow('dst', img1_orig)

Ix = cv2.Sobel(img1, cv2.CV_32F, 1, 0, ksize=3)
Iy = cv2.Sobel(img1, cv2.CV_32F, 0, 1, ksize=3)
Ix2 = np.matrix(Ix * Ix, dtype=np.float32)
Iy2 = np.matrix(Iy * Iy, dtype=np.float32)
IxIy = np.matrix(Ix * Iy, dtype=np.float32)
print(f'Ix2.max(): {Ix2.max()}')
print(f'Iy2.max(): {Iy2.max()}')
print(f'IxIy.max(): {IxIy.max()}')

def detectFeatures(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            det = Ix2[i, j] * Iy2[i, j] - (IxIy[i, j])**2
            trace = Ix2[i, j] + Iy2[i, j]
            R = det / trace
            if not math.isnan(R) and R > 0:
                # TODO: Apply local maximum
                img[i, j] = [0, 0, 255]
    return img

features = detectFeatures(img1_orig.copy())
cv2.imshow('features', features)
# Ix2sum, Iy2sum, IxIysum = Ix2.sum(), Iy2.sum(), IxIy.sum()
# det = Ix2sum*Iy2sum - (IxIysum)**2
# tr = Ix2sum + Iy2sum
# print('det: ' + str(det))
# print('tr: ' + str(tr))
# # Corner response function
# R = det / tr
# print('R: ' + str(R))
cv2.waitKey(0)
cv2.destroyAllWindows()
