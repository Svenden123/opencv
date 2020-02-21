from pyimagesearch.transform import four_point_transform
import argparse
import cv2
import imutils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)
args = vars(ap.parse_args())

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
fullPath = args["image"]
partedPath = fullPath.split('.')
prePath = partedPath[0]
ext = partedPath[-1]
image = cv2.imread(fullPath)
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

# scale contrast and blur
gray = cv2.convertScaleAbs(image, alpha=1.2, beta=0)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break

# draw circle on angles
extLeft = tuple(screenCnt[0][0])
extRight = tuple(screenCnt[1][0])
extTop = tuple(screenCnt[2][0])
extBot = tuple(screenCnt[3][0])
angled = image.copy()
cv2.circle(angled, extLeft, 8, (0, 255, 0), -1)
cv2.circle(angled, extRight, 8, (0, 255, 0), -1)
cv2.circle(angled, extTop, 8, (0, 255, 0), -1)
cv2.circle(angled, extBot, 8, (0, 255, 0), -1)

# apply the four point transform to obtain a top-down
# view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# save the angled and scanned images
cv2.imwrite(prePath+'_angled.'+ext, angled)
cv2.imwrite(prePath+'_warped.'+ext, warped)

#cv2.imshow("orig", orig)
#cv2.imshow("gray", gray)
#cv2.imshow("edged", edged)
#cv2.imshow("angled", angled)
#cv2.imshow("warped", warped)
#cv2.waitKey(0)
#cv2.destroyAllWindows()