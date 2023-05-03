import os
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.spatial import distance
import numpy as np

DirPath1 = '../GallerySet'
DirPath2 = '../ProbeSet'
print(DirPath2)

Files_G = os.listdir(DirPath1)
Files_P = os.listdir(DirPath2)
count = 0
binary_g = []
binary_p = []

for File1 in Files_G:
    imgPath1 = os.path.join(DirPath1, File1)
    # print(imgPath1)
    image_G = cv.imread(imgPath1, cv.IMREAD_GRAYSCALE)
    threshold, binary_img_g = cv.threshold(image_G, 128, 1, cv.THRESH_BINARY)
    # cv.imwrite(f'C://Users//somap//Workspace//Hw-CV//Project//binary_G//bin_{File1}', binary_img_g)
    binary_g.append(binary_img_g.flatten())
    # plt.gray()
    # plt.title("Binary image of faces from Gallery")
    # plt.imshow(binary_img)
    # plt.show()
    count = count + 1
print(len(binary_g))

for File2 in Files_P:
    imgPath2 = os.path.join(DirPath2, File2)
    # print(imgPath2)
    image_P = cv.imread(imgPath2, cv.IMREAD_GRAYSCALE)
    threshold, binary_img_p = cv.threshold(image_P, 128, 1, cv.THRESH_BINARY)
    # cv.imwrite(f'C://Users//somap//Workspace//Hw-CV//Project//binary_P//bin_{File2}', binary_img_p)
    binary_p.append(binary_img_p.flatten())
    # plt.gray()
    # plt.title("Binary image of faces from Probe")
    # plt.imshow(binary_img_p)
    # plt.show()
# print(binary_p)
# Need to calculate hamming distance and generate score matrix
# score_matrix =np.full(100, 255, dtype=int)
# score_matrix = np.array([], [])
score_matrix = np.zeros((100, 100))

print('a', len(binary_g[99]))
print('b', len(binary_p[99]))
# print(binary_p.any())

for r in range(len(binary_g)):
    for c in range(len(binary_p)):
        ham_distance = (distance.hamming(binary_g[r], binary_p[c]))
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
# print(genuine.size)

d = np.average(imposter)
c = np.average(genuine)
p = np.std(imposter)
q = np.std(genuine)
d = (2 ** .5 * abs(d - c) / (p ** 2 + q ** 2) ** .5)

print("Decidibility index value", d)
