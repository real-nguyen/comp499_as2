# Note that this assignment uses opencv-python 3.4.5.20 due to a bug where drawKeypoints is not available in the latest version
import cv2
import numpy as np
import os
import math

THRESHOLD = 0.00025
SSD_THRESHOLD = 350
# For SIFT descriptor
WINDOW_SIZE = 16
CELL_SIZE = 4
DIRNAME = os.path.dirname(__file__)
DIR_YOSEMITE = os.path.join(DIRNAME, 'yosemite')
yosemite1 = os.path.join(DIR_YOSEMITE, 'Yosemite1.jpg')
yosemite2 = os.path.join(DIR_YOSEMITE, 'Yosemite2.jpg')
img1_orig = cv2.imread(yosemite1)
img2_orig = cv2.imread(yosemite2)
img1 = cv2.imread(yosemite1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(yosemite2, cv2.IMREAD_GRAYSCALE)

def detectFeatures(img, Ix, Iy):
    Ix2 = np.matrix(Ix * Ix, dtype=np.float32)
    Iy2 = np.matrix(Iy * Iy, dtype=np.float32)
    IxIy = np.matrix(Ix * Iy, dtype=np.float32)
    keypoints = img.copy()
    keypoints[:] = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            det = Ix2[i, j] * Iy2[i, j] - (IxIy[i, j])**2
            trace = Ix2[i, j] + Iy2[i, j]
            R = det / trace if trace != 0 else 0
            if not math.isnan(R) and R > THRESHOLD:
                keypoints[i, j] = img[i, j]
    local_max = get_local_max(keypoints)
    return local_max

def get_local_max(img):
    # Returns a black image with dots representing the features
    local_max = img.copy()
    local_max[:] = 0
    for i in range(img.shape[0]):        
        for j in range(img.shape[1]):
            # Construct 3x3 window around pixel (i, j)
            # If pixel is in a corner or at an edge, pad with 0's
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
            # Keep centre pixel if it's the max in its window, otherwise suppress it
            local_max[i,j] = img[i,j] if img[i,j] == padded_img.max() else 0
    return local_max

def get_keypoints(features):
    keypoints = []
    for y in range(features.shape[0]):
        for x in range(features.shape[1]):
            if features[y, x] > 0:
                keypoints.append(cv2.KeyPoint(x, y, 1))
    return keypoints

def SIFT(img, Ix, Iy, features):
    nz = np.nonzero(features)
    feature_coords = list(zip(*nz))
    orientations = get_orientations(Ix, Iy)
    descriptors = []
    for coord in feature_coords:
        window = get_window(orientations, coord, img.shape)
        # Window goes out of bounds; ignore this feature
        if window is None:
            continue
        cells = get_cells(window)
        # Raw descriptor
        descriptor = get_descriptor(cells)        
        magnitude = math.sqrt(np.sum(np.square(descriptor)))
        # Normalized descriptor
        # Will equal ~1 when squared and summed up
        descriptor = [d / magnitude for d in descriptor]
        descriptors.append(descriptor)
    return descriptors
        

def get_window(orientations, feature_coord, img_shape):
    # Get top left and bottom right corners of window based off of feature coordinates
    # The feature coordinates correspond to the approximate centre of the window
    half_window = int(WINDOW_SIZE/2)
    centre_y, centre_x = feature_coord[0], feature_coord[1]
    up_left_y = centre_y - half_window
    up_left_x = centre_x - half_window
    # If window goes out of the image's bounds, throw it away
    if up_left_y < 0 or up_left_x < 0:
        return None
    bottom_right_y = centre_y + half_window
    bottom_right_x = centre_x + half_window
    if bottom_right_y >= img_shape[0] or bottom_right_x >= img_shape[1]:
        return None
    # Dump orientations from this area into window, starting with upper left corner
    window = []
    # Add 1's to ranges to get correct centre and window dimensions
    for row in range(up_left_y + 1, bottom_right_y + 1):
        temp_row = []
        for col in range(up_left_x + 1, bottom_right_x + 1):
            temp_row.append(orientations[row, col])
        window.append(temp_row)
    return np.matrix(window)

def get_cells(window):
    row_idx = 0
    col_idx = 0
    cells = []
    cell = []
    while row_idx < WINDOW_SIZE:    
        row = []
        while col_idx < WINDOW_SIZE:
            row.append(window[row_idx,col_idx])
            if (col_idx+1) % CELL_SIZE == 0:
                break
            col_idx += 1
        cell.append(row)
        if row_idx == WINDOW_SIZE - 1 and col_idx == WINDOW_SIZE - 1:
            # Last cell in the window
            cells.append(cell)
            break
        elif (row_idx+1) % CELL_SIZE == 0 and col_idx == WINDOW_SIZE - 1:
            # Last cell on the right
            # Wrap back col to left of window and shift row down
            cells.append(cell)
            cell = []
            row_idx += 1
            col_idx = 0
        elif (row_idx+1) % CELL_SIZE == 0:
            # Last row in cell
            # Wrap back row to top of cell and shift col right
            cells.append(cell)
            cell = []
            row_idx -= (CELL_SIZE-1)
            col_idx += 1
        else:
            # Cell row complete
            # Wrap back col to left of cell and shift row down
            row_idx += 1
            col_idx -= (CELL_SIZE-1)
    return cells

def get_orientations(Ix, Iy):
    orientations = Ix.copy()
    orientations[:] = 0
    for row in range(Ix.shape[0]):
        for col in range(Ix.shape[1]):
            # arctan goes from -pi/2 to pi/2 rads (-90 to 90 degrees)
            orientations[row, col] = math.atan(Iy[row,col] / Ix[row,col]) if Ix[row,col] != 0 else 0
    return orientations

def get_descriptor(cells):
    descriptor = []
    for cell in cells:
        # Array with fixed size 8
        histogram = [0] * 8
        cell_flat = [item for row in cell for item in row]
        pi = math.pi
        for orientation in cell_flat:
            if 0 <= orientation < pi/8:
                histogram[0] += 1
            elif pi/8 <= orientation < pi/4:
                histogram[1] += 1
            elif pi/4 <= orientation < 3*pi/8:
                histogram[2] += 1
            elif 3*pi/8 <= orientation < pi/2:
                histogram[3] += 1
            elif -pi/8 <= orientation < 0:
                histogram[4] += 1
            elif -pi/4 <= orientation < -pi/8:
                histogram[5] += 1
            elif -3*pi/8 <= orientation < -pi/4:
                histogram[6] += 1
            elif -pi/2 < orientation < -3*pi/8:
                histogram[7] += 1
        # 16 cells * 8 orientations = 128 elements total in descriptor
        descriptor += histogram[:]
    return descriptor

def match_features(img1_descriptors, img2_descriptors):
    for img1_feature in img1_descriptors:
        distances = []
        for img2_feature in img2_descriptors:
            ssd = sum((np.array(img1_feature) - np.array(img2_feature))**2)
            if ssd <= SSD_THRESHOLD:
                distances.append(ssd)
        if len(distances) == 0 or len(distances) == 1:
            continue
        distances.sort()
        print(f'Lowest SSD for feature {img1_descriptors.index(img1_feature)}: {distances[0]}; Ratio test: {distances[0]/distances[1]}')
        
        
# Get image gradients
img1_Ix = cv2.Sobel(img1, cv2.CV_32F, 1, 0, ksize=3)
img1_Iy = cv2.Sobel(img1, cv2.CV_32F, 0, 1, ksize=3)
img2_Ix = cv2.Sobel(img2, cv2.CV_32F, 1, 0, ksize=3)
img2_Iy = cv2.Sobel(img2, cv2.CV_32F, 0, 1, ksize=3)

img1_features = detectFeatures(img1.copy(), img1_Ix, img1_Iy)
img1_keypoints = get_keypoints(img1_features)
cv2.imshow('img1_features', img1_features)
img1_display_keypoints = img1_orig.copy()
cv2.drawKeypoints(img1_orig, img1_keypoints, img1_display_keypoints)
cv2.imshow('img1_orig', img1_orig)
cv2.imshow('img1_display_keypoints', img1_display_keypoints)
img1_descriptors = SIFT(img1, img1_Ix, img1_Iy, img1_features)
print(f'Number of descriptors for Yosemite1.jpg: {len(img1_descriptors)}')

img2_features = detectFeatures(img2.copy(), img2_Ix, img2_Iy)
img2_keypoints = get_keypoints(img2_features)
cv2.imshow('img2_features', img2_features)
img2_display_keypoints = img2_orig.copy()
cv2.drawKeypoints(img2_orig, img2_keypoints, img2_display_keypoints)
cv2.imshow('img2_orig', img2_orig)
cv2.imshow('img2_display_keypoints', img2_display_keypoints)
img2_descriptors = SIFT(img2, img2_Ix, img2_Iy, img2_features)
print(f'Number of descriptors for Yosemite2.jpg: {len(img2_descriptors)}')

match_features(img1_descriptors, img2_descriptors)

cv2.waitKey(0)
cv2.destroyAllWindows()
