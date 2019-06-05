import cv2
import numpy as np

def nothing(x):
    pass

def circular_color_mask(image, hsv_range):

	result = image
	low_h, low_s, low_v = hsv_range[0]
	high_h, high_s, high_v = hsv_range[1]

	if (hsv_range[0]<=hsv_range[1]).all():
		lower_boundary = np.array((low_h, low_s, low_v), dtype = "uint8")
		upper_boundary = np.array((high_h, high_s, high_v), dtype = "uint8")
		result = cv2.inRange(image, lower_boundary, upper_boundary)
	else:
		lower_boundary = np.array((low_h, low_s, low_v), dtype = "uint8")
		upper_boundary = np.array((low_h > high_h and 179 or high_h, low_s > high_s and 255 or high_s, low_v > high_v and 255 or high_v), dtype = "uint8")
		lower_boundary2 = np.array((low_h <= high_h and low_h or 0, low_s <= high_s and low_s or 0, low_v <= high_v and low_v or 0), dtype = "uint8")
		upper_boundary2 = np.array((high_h, high_s, high_v), dtype = "uint8")
		result = cv2.bitwise_or(cv2.inRange(image, lower_boundary, upper_boundary), cv2.inRange(image, lower_boundary2, upper_boundary2))

	return result

def thresholdImage(image, args):
    """ Performs thresholding on an image
    Parameters
    ----------
    image : numpy.ndarray
        the image to process
    args : list
        a list of additional arguments
    Returns
    --------
        numpy.ndarray
            the processed image
    """
    thresh = args[0]
    pix = args[1]
    algo=cv2.THRESH_BINARY
    result = cv2.threshold(image, thresh, pix, algo)[1]
    return result


# Create a black image, a window
origin = cv2.imread('FVR.png')
hsv = cv2.cvtColor(origin, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
img = circular_color_mask(hsv, np.array([[50, 120, 120], [130, 255, 255]]))
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('Low-H','image',0,179,nothing)
cv2.createTrackbar('High-H','image',0,179,nothing)
cv2.createTrackbar('Low-S','image',0,255,nothing)
cv2.createTrackbar('High-S','image',0,255,nothing)
cv2.createTrackbar('Low-V','image',0,255,nothing)
cv2.createTrackbar('High-V','image',0,255,nothing)

# create switch for ON/OFF functionality
switch = 'FLIP:\n0 : COLOR \n1 : THRESHOLD \n2 : REVERSE_THRESHOLD'
cv2.createTrackbar(switch, 'image',0,2,nothing)

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

    args = np.array([[low_h, low_s, low_v],[high_h, high_s, high_v]])

    if flip == 1:
        img = thresholdImage(gray, [low_v, high_v])
    elif flip == 2:
        img = cv2.bitwise_not(thresholdImage(gray, [low_v, high_v]))
    else:
        img = circular_color_mask(hsv, args)

cv2.destroyAllWindows()
