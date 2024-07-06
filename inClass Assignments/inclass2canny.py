import cv2 
import numpy as np

def solve(mystring):
    image = cv2.imread(mystring)
    image = cv2.resize(image, (image.shape[0]//2, image.shape[1]//2))
    image_original = image 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.Canny(image, 170, 200)

    cv2.imshow("Final", image)
    cv2.imshow("original", image_original)
    cv2.waitKey()

solve("cv/cvproj/image for mideval.jpg")