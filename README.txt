Constants:
THRESHOLD: Corner response function threshold. Change this to get a different number of keypoints. Should be somewhere between 0 and 0.001. Default value = 0.00025
SSD_THRESHOLD: Squared sum distance threshold for feature matching. The lower this number is, the less matches there will be. Default value = 350
WINDOW_SIZE: The size of the SIFT descriptor window. The window should always be a square with even dimensions. Default value = 16
CELL_SIZE: The size of the cells that divide the SIFT descriptor window. Should always be a square with even dimensions. Default value = 4

Methods are fairly self-explanatory by their names and comments.