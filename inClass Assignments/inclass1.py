import cv2
import numpy as np
import matplotlib.pyplot as plt

def solve(mystring):
    image = cv2.imread(mystring)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = np.fft.fft2(image_gray)
    fft_shifted = np.fft.fftshift(result)
    result = fft_shifted
    magnitude_spectrum = np.abs(result)
    result = magnitude_spectrum
    cv2.imshow("result", result)
    cv2.waitKey()

solve("cv/cvproj/image for mideval.jpg")