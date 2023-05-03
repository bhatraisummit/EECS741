import cv2
import matplotlib.pyplot as plt
import numpy as np

# reading image
img1 = cv2.imread('../GallerySet/subject2_img1.pgm', cv2.IMREAD_GRAYSCALE)

# keypoints
sift = cv2.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
for keypoint in keypoints_1:
    print(keypoint.pt)
print(descriptors_1.shape)

img_1 = cv2.drawKeypoints(img1, keypoints_1, img1)
# cv2.imwrite('sift.jpg', img_1)

img1 = cv2.imread('../ProbeSet/subject2_img2.pgm', cv2.IMREAD_GRAYSCALE)

# keypoints
sift = cv2.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
print(descriptors_1[0])

img_1 = cv2.drawKeypoints(img1, keypoints_1, img1)
# cv2.imwrite('sift2.jpg', img_1)

