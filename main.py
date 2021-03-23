import cv2

img = cv2.imread('ISBI 2012/labels/train-labels00.jpg', cv2.IMREAD_UNCHANGED)
print(img.shape)
print(img)