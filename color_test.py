import cv2
import numpy as np

def nothing(x):
    pass

def circular_color_mask(image, low_h, low_s, low_v, high_h, high_s, high_v):

    result = image

    if low_h <= high_h and low_s <= high_s and low_v <= high_v:
        lower_boundary = np.array((low_h, low_s, low_v), dtype = "uint8")
        upper_boundary = np.array((high_h, high_s, high_v), dtype = "uint8")
        result = cv2.inRange(image, lower_boundary, upper_boundary)
    else:
        lower_boundary = np.array((low_h, low_s, low_v), dtype = "uint8")
        upper_boundary = np.array((high_h >= low_h and high_h or 179, high_s >= low_s and high_s or 255, high_v >= low_v and high_v or 255), dtype = "uint8")
        lower_boundary2 = np.array((low_h >= high_h and 1 or low_h, low_s >= high_s and 1 or low_s, low_v >= high_v and 1 or low_v), dtype = "uint8")
        upper_boundary2 = np.array((high_h, high_s, high_v), dtype = "uint8")
        result = cv2.bitwise_or(cv2.inRange(image, lower_boundary, upper_boundary), cv2.inRange(image, lower_boundary2, upper_boundary2))

    return result


# Create a black image, a window
origin = cv2.imread('test.png')
hsv = cv2.cvtColor(origin, cv2.COLOR_RGB2HSV)
img = circular_color_mask(hsv, 50, 120, 120, 130, 255, 255)
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('Low-H','image',0,179,nothing)
cv2.createTrackbar('High-H','image',0,179,nothing)
cv2.createTrackbar('Low-S','image',0,255,nothing)
cv2.createTrackbar('High-S','image',0,255,nothing)
cv2.createTrackbar('Low-V','image',0,255,nothing)
cv2.createTrackbar('High-V','image',0,255,nothing)

# create switch for ON/OFF functionality
switch = 'FLIP:\n0 : NONE \n1 : HUE \n2 : SATURATION \n3 : VALUE \n4 : EVERYTHING'
cv2.createTrackbar(switch, 'image',0,4,nothing)

while(1):
    cv2.imshow('image',img)
    cv2.imshow('origin',origin)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    low_h = cv2.getTrackbarPos('Low-H','image')
    low_s = cv2.getTrackbarPos('Low-S','image')
    low_v = cv2.getTrackbarPos('Low-V','image')
    high_h = cv2.getTrackbarPos('High-H','image')
    high_s = cv2.getTrackbarPos('High-S','image')
    high_v = cv2.getTrackbarPos('High-V','image')
    flip = cv2.getTrackbarPos(switch,'image')

    if flip == 1:
        img = circular_color_mask(hsv, high_h, low_s, low_v, low_h, high_s, high_v)
    elif flip == 2:
        img = circular_color_mask(hsv, low_h, high_s, low_v, high_h, low_s, high_v)
    elif flip == 3:
        img = circular_color_mask(hsv, low_h, low_s, high_v, high_h, high_s, low_v)
    elif flip == 4:
        img = circular_color_mask(hsv, high_h, high_s, high_v, low_h, low_s, low_v)
    else:
        img = circular_color_mask(hsv, low_h, low_s, low_v, high_h, high_s, high_v)

cv2.destroyAllWindows()