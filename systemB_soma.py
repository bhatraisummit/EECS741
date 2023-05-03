import os
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.spatial import distance
import numpy as np
from skimage.filters import threshold_sauvola
from sklearn.metrics import jaccard_score

DirPath1 = '../GallerySet'
DirPath2 = '../ProbeSet'
print(DirPath2)


def contrast_stretch(img):
    # Map the intensity values between the thresholds to the full range of pixel values
    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    img = cv.LUT(img, table)
    return img

def manhattan_distance(point1, point2):
    return sum(abs(value1 - value2) for value1, value2 in zip(point1, point2))


Files_G = os.listdir(DirPath1)
Files_P = os.listdir(DirPath2)

def sorted(g):
    g1 = np.zeros(100, dtype=object)

    for i in range(0, len(g)):
      a = int(g[i].split("_")[0].split("t")[1])-1
      g1[a] = g[i]
    return g1

Files_G = sorted(Files_G)
Files_P = sorted(Files_P)


# print(Files_P)
count = 0
binary_g = []
binary_p = []

for File1 in Files_G:
    imgPath1 = os.path.join(DirPath1, File1)
    image_G = cv.imread(imgPath1, cv.IMREAD_GRAYSCALE)
    median = cv.medianBlur(image_G, 3)
    median = cv.medianBlur(median, 3)
    median = cv.medianBlur(median, 3)
    image_G = cv.addWeighted(image_G, 7.5, median, -6.5, 17)
    # image_G = contrast_stretch(image_G)
    binary_img_g = cv.adaptiveThreshold(image_G,
                                        1,
                                        cv.ADAPTIVE_THRESH_MEAN_C,
                                        cv.THRESH_BINARY,
                                        31,
                                        10)

    binary_g.append(binary_img_g)
    count = count + 1
print(len(binary_g))

for File2 in Files_P:
    imgPath2 = os.path.join(DirPath2, File2)
    image_P = cv.imread(imgPath2, cv.IMREAD_GRAYSCALE)
    median2 = cv.medianBlur(image_P, 3)
    median2 = cv.medianBlur(median2, 3)
    median2 = cv.medianBlur(median2, 3)
    # image_P = contrast_stretch(image_P)
    image_P = cv.addWeighted(image_P, 7.5, median2, -6.5, 17)

    binary_img_p = cv.adaptiveThreshold(image_P,
                                        1,
                                        cv.ADAPTIVE_THRESH_MEAN_C,
                                        cv.THRESH_BINARY,
                                        31,
                                        10)

    binary_p.append(binary_img_p)
score_matrix = np.zeros((100, 100))

for r in range(len(binary_g)):
    for c in range(len(binary_p)):
        ham_distance = (jaccard_score(binary_g[r], binary_p[c], average='samples'))
        score_matrix[r][c] = ham_distance
print(score_matrix)

# Now to create imposter and Genuin score and decidability factor

genuine = []
imposter = []

for x in range(100):
    for y in range(100):
        if y == x:
            genuine.append(score_matrix[x][y])
        else:
            imposter.append(score_matrix[x][y])

imposter = np.array(imposter)
# print(imposter.size)

gens = np.array(genuine)
print(gens)
# print(genuine.size)

d = np.average(imposter)
c = np.average(genuine)
p = np.std(imposter)
q = np.std(genuine)
d = (2 ** .5 * abs(d - c) / (p ** 2 + q ** 2) ** .5)

print("Decidibility index value", d)














