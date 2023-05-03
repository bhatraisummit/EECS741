import cv2
import numpy as np
import matplotlib.pyplot as plt

# read image as grayscale float in range 0 to 1
img = cv2.imread('../GallerySet/subject2_img1.pgm', cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0

# set arguments
gamma = 0.5
alpha = 0.1
tau = 3.0

# gamma correction
img_gamma = np.power(img, gamma)
# img_gamma = img
img_gamma2 = (255.0 * img_gamma).clip(0, 255).astype(np.uint8)

# DOG
blur1 = cv2.GaussianBlur(img_gamma, (0, 0), 1, borderType=cv2.BORDER_REPLICATE)
blur2 = cv2.GaussianBlur(img_gamma, (0, 0), 3, borderType=cv2.BORDER_REPLICATE)
blur3 = cv2.GaussianBlur(img_gamma, (0, 0), 3, borderType=cv2.BORDER_REPLICATE)
img_dog = (blur1 - blur2)
# img_dog = (img_dog - blur3)
# normalize by the largest absolute value so range is -1 to
img_dog = img_dog / np.amax(np.abs(img_dog))
img_dog2 = (255.0 * (0.5 * img_dog + 0.5)).clip(0, 255).astype(np.uint8)
ret, th1 = cv2.threshold(img_dog2, 120, 255, cv2.THRESH_BINARY)

# contrast equalization equation 1
img_contrast1 = np.abs(img_dog)
img_contrast1 = np.power(img_contrast1, alpha)
img_contrast1 = np.mean(img_contrast1)
img_contrast1 = np.power(img_contrast1, 1.0 / alpha)
img_contrast1 = img_dog / img_contrast1

# contrast equalization equation 2
img_contrast2 = np.abs(img_contrast1)
img_contrast2 = img_contrast2.clip(0, tau)
img_contrast2 = np.mean(img_contrast2)
img_contrast2 = np.power(img_contrast2, 1.0 / alpha)
img_contrast2 = img_contrast1 / img_contrast2
img_contrast = tau * np.tanh((img_contrast2 / tau))

# Scale results two ways back to uint8 in the range 0 to 255
img_contrastA = (255.0 * (img_contrast + 0.5)).clip(0, 255).astype(np.uint8)
img_contrastB = (255.0 * (0.5 * img_contrast + 0.5)).clip(0, 255).astype(np.uint8)

# show results
titles = ['Original Image', 'Gamma',
          'DoG', 'CE1',
          'CE_A', 'CE_B']
images = [img, img_gamma2, img_dog2, img_contrast1, img_contrastA, img_contrastB]
for i in range(6):
    plt.subplot(3, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
