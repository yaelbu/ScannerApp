import imutils as imutils
from imutils.perspective import four_point_transform
import cv2
import numpy as np
import sys

#STEP 1- Detect edges in document
path_input_image=sys.argv[1]
path_output_image=sys.argv[2]
image = cv2.imread(path_input_image)
#image = cv2.imread("page.jpg")
#image = cv2.imread("Game.jpg")
#image = cv2.imread("Receipt.jpg")
if image is None:
	print('Error opening image!')
ratio = image.shape[0] / 550.0
original_image = image.copy()
image = imutils.resize(image, height = 550)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image= cv2.GaussianBlur(gray_image, (5, 5), 0)
image_with_edges = cv2.Canny(gray_image, 100, 150)
print("STEP 1: Edge Detection")
cv2.imshow("STEP1-edge", image_with_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

#STEP 2- Finding the contour of the scanned document
image_contours = cv2.findContours(image_with_edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
image_contours = imutils.grab_contours(image_contours)
image_contours = sorted(image_contours, key = cv2.contourArea, reverse = True)[:5]
for i in image_contours:
	perimeter = cv2.arcLength(i, True) # approximate the contour
	epsilon=0.02 * perimeter
	approx = cv2.approxPolyDP(i, epsilon, True)
	if len(approx) == 4: #if the approximated contour has 4 points then we have a document
		document_contours = approx
		break
print("STEP 2: Contours of document")
cv2.drawContours(image, [document_contours], -1, (0, 255, 0), 2)
cv2.imshow("STEP2-contour", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#STEP 3- Apply a transformation on the page to get a "top-down" document view
transformed_image = four_point_transform(original_image, document_contours.reshape(4, 2) * ratio)

#STEP 4-Perform binarization to obtain a black-and-white scan
final_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)
thres, final_image = cv2.threshold(final_image, 95, 255, cv2.THRESH_BINARY)
print("STEP 3+4: Apply perspective transform and convert to black and white")
cv2.imwrite(path_output_image,final_image)
cv2.imshow("STEP3+4", imutils.resize(final_image, height = 550))
cv2.waitKey(0)
