import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image 

test_img = cv.imread("cv/WIN_20231017_23_15_57_Pro - frame at 0m17s.jpg")
# cv.imshow('image', test_img)
# cv.waitKey()
# cv.destroyAllWindows()

# test_img2 = cv.imread("cv/WIN_20231017_23_20_12_Pro - frame at 1m9s.jpg")
# cv.imshow('umage2', test_img2)
# cv.waitKey()  #this was a partial image
# cv.destroyAllWindows()

# test_img3 = cv.imread("cv/WIN_20231017_23_22_40_Pro - frame at 0m3s.jpg")
# cv.imshow('umage3', test_img3)
# cv.waitKey()
def imshow(str, img = None):
    cv.imshow(str, img)
    cv.waitKey()
    cv.destroyAllWindows()

def lowerTheBlue(img, index):
    b,g,r = cv.split(img)
    blue_low = cv.merge([cv.subtract(b, np.ones((img.shape[0],img.shape[1]), dtype= img.dtype)*index), g, r])
    return blue_low

img=test_img

kernel = np.ones((2,2), dtype= img.dtype)

dilate = cv.dilate(img, kernel, iterations=10)
eroded = cv.erode(img, kernel, iterations=10)
imshow('none',dilate)
# imshow('nani',eroded)

openned = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

# imshow('openned', openned)
# imshow('org', img)

canny = cv.Canny(img, 0, 40)
imshow('canneid', canny)
imshow('o', img)

