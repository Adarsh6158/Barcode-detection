import os
import argparse
from pyzbar import pyzbar
import numpy as np
import cv2

def preprocess(image_path):
    # load the image
    image = cv2.imread(image_path)

    # resize image
    image = cv2.resize(image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # calculate x & y gradient
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # blur the image
    blurred = cv2.blur(gradient, (3, 3))

    # threshold the image
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    thresh = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return thresh

def barcode(image_path):
    # load the image
    image = cv2.imread(image_path, 0)

    # scan the image for barcodes
    barcodes = pyzbar.decode(image)

    # extract results
    for barcode in barcodes:
        barcode_data = barcode.data.decode("utf-8")
        print(f'Format: {barcode.type}, Data: {barcode_data}')
        print('---------------------------------')
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image file")
args = vars(ap.parse_args())

image_path = args["image"]
image = preprocess(image_path)
barcode(image_path)
