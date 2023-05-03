import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import math
import cv2
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from skimage import metrics
from skimage import filters, exposure
from skimage.filters import threshold_local
from skimage.feature import local_binary_pattern
from skimage import feature


# using the threshold of 128 for binarization
def thres_(img):
    # img = Image.open('image.jpg').convert('L')

    # Convert image to numpy array
    img_arr = np.asarray(img)

    # Apply threshold
    thresh_value = 128  # Change this value to adjust the threshold
    thresh_arr = np.where(img_arr > thresh_value, 1, 0)
    return thresh_arr


############################-----------------------------------------------------
data_dir = '../GallerySet'

data_dir1 = '../ProbeSet'
g = os.listdir(data_dir)
p = os.listdir(data_dir1)


############################-----------------------------------------------------

# structural similarity

def ssim(img1: np.ndarray, img2: np.ndarray):
    img1 = img1.flatten()
    img2 = img2.flatten()
    return metrics.structural_similarity(img1, img2)


# cosine similarity
def cosine_sim(img1: np.ndarray, img2: np.ndarray):
    # img = np.asarray(img)

    img1 = img1.flatten()
    img2 = img2.flatten()
    img1 = img1.reshape(1, -1)
    img2 = img2.reshape(1, -1)
    cosim = cosine_similarity(img1, img2)

    return cosim


# hamming distance
def hamming_(img1: np.ndarray, img2: np.ndarray):
    image1 = np.ravel(img1)
    # image1 = img1.flatten()
    # print(image1)
    image2 = np.ravel(img2)
    dist = distance.hamming(image1, image2)
    return dist


# normalized correlation
def normalized_corr(img1: np.ndarray, img2: np.ndarray):
    x = img1.reshape((-1, 1))
    y = img2.reshape((-1, 1))
    xn = x - np.mean(x)
    yn = y - np.mean(y)
    r = (np.sum(xn * yn)) / (np.sqrt(np.sum(xn ** 2)) * np.sqrt(np.sum(yn ** 2)))
    return r


# print(p)


def sorted(g):
    g1 = np.zeros(100, dtype=object)
    for i in range(0, len(g)):
        a = int(g[i].split("_")[0].split("t")[1]) - 1
        g1[a] = g[i]
    return g1


# print(p)
g = sorted(g)
p = sorted(p)
# print(p)

r = np.zeros([100, 100], dtype=np.float64)
probe = []
gallery = []
[gallery.append(cv2.imread(f"{data_dir}/{i}", cv2.IMREAD_GRAYSCALE)) for i in g]
[probe.append(cv2.imread(f"{data_dir1}/{i}", cv2.IMREAD_GRAYSCALE)) for i in p]


# [gallery.append(Image.open(f"{data_dir}/{i}").convert('L')) for i in g]
# [probe.append(Image.open(f"{data_dir1}/{i}").convert('L')) for i in p]

# print(gallery)

###### using thresholding and distance measure -----------------------------------

def thresh_hamming(probe, gallery, hamming_):
    for j in range(0, len(probe)):
        for i in range(0, len(gallery)):
            r[j, i] = hamming_(thres_(probe[j]), thres_(gallery[i]))
    print(r.shape)
    print(r[0:9, 0:9])
    return r


###### using otsu thresholding and distance measure -----------------------------------
def otsu_1(img):
    img_arr = np.asarray(img)
    threshold = filters.threshold_otsu(img_arr)
    thresholded = img > threshold
    # print(thresholded)
    # binarized_img = np.where(thresholded > 0, 1, 0)

    return thresholded


def otsu_(probe, gallery, hamming_):
    def otsu_(img):
        img = cv2.GaussianBlur(img, (5, 5), 1)

        img_arr = contrast_stretch(img)

        value, thresh = cv2.threshold(img_arr, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def otsu_1(img):
        img = cv2.GaussianBlur(img, (5, 5), 1)

        img_arr = contrast_stretch(img)
        threshold = filters.threshold_otsu(img_arr)
        thresholded = img > threshold
        # print(thresholded)
        # binarized_img = np.where(thresholded > 0, 1, 0)

        return thresholded

    for j in range(0, len(probe)):
        for i in range(0, len(gallery)):
            r[j, i] = hamming_(otsu_1(probe[j]), otsu_1(gallery[i]))
    # print(r.shape)
    # print(r[0:9, 0:9])
    return r


def adaptive_thres(probe, gallery, alg, x, y):
    def adaptive_(img, x, y):
        # img = np.asarray(img)
        img = cv2.medianBlur(img, 5)
        # img = contrast_stretch(img)
        thres_ = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, x, y)
        return thres_

    for j in range(0, len(probe)):
        for i in range(0, len(gallery)):
            r[j, i] = alg(adaptive_(probe[j], x, y), adaptive_(gallery[i], x, y))
    # print(r.shape)
    # print(r[0:9, 0:9])
    return r


def thres_local(probe, gallery, alg, x, y):
    def thres_local_(img):
        # img =np.asarray(img)
        # img = cv2.GaussianBlur(img, (5,5), 0)
        # img = cv2.medianBlur(img, 5)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # img = contrast_stretch(img)
        # img = contrast_stretch(img)

        thres = threshold_local(img, block_size=x, method='gaussian', offset=y)
        binarized = (img > thres).astype('uint8')
        # print(binarized)
        # exit()
        return binarized

    for j in range(0, len(probe)):
        for i in range(0, len(gallery)):
            r[j, i] = alg(thres_local_(probe[j]), thres_local_(gallery[i]))
    return r


def lbp(probe, gallery, alg, x, y, bins):
    # pass
    def lbp_(img):
        # img = np.asarray(img)
        img = cv2.medianBlur(img, 5)

        img = contrast_stretch(img)

        pattern = local_binary_pattern(img, x, y, method='uniform')
        n_bins = int(img.max() + 1)
        # print(n_bins)
        # n_bins = 1
        hist_, _ = np.histogram(pattern, density=True, bins=n_bins, range=(0, n_bins))
        # hist_ = hist_.reshape(-1)
        # cdf = hist_.cumsum()
        # cdf_normalized = cdf * hist_.max() / cdf.max()
        # hist_ = cdf_normalized
        thres_ = np.argmax(pattern)
        binarized = (img > thres_).astype('uint8')
        return binarized

    for j in range(0, len(probe)):
        for i in range(0, len(gallery)):
            r[j, i] = alg(lbp_(probe[j]), lbp_(gallery[i]))
    return r


def contrast_stretch(img):
    # Map the intensity values between the thresholds to the full range of pixel values
    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    img = cv2.LUT(img, table)
    # print(img)
    # exit()
    return img


def main(r):
    genuine_scores = []
    imposter_scores = []

    def get_gen(rr):
        genuine_scores = []

        for i in range(0, len(rr)):
            genuine_scores.append(rr[i][i])
        return genuine_scores

    g = np.asarray(get_gen(r))

    a = np.eye(r.shape[0], dtype=bool)
    impostor_scores = r[~np.eye(r.shape[0], dtype=bool)]

    # decidability index
    def decidability_(genuine_scores, imposter_scores):
        mean_1 = np.mean(genuine_scores)
        mean_0 = np.mean(imposter_scores)
        s_1 = np.std(genuine_scores)
        s_0 = np.std(imposter_scores)
        decidability = math.sqrt(2) * abs(mean_1 - mean_0) / math.sqrt(s_1 ** 2 + s_0 ** 2)

        return decidability

    # print(genuine_scores)
    dec_ = decidability_(g, impostor_scores)
    return dec_
    # print(dec_)
    # print


# r = thresh_hamming(probe,gallery,hamming_)
# main(r)
# r = otsu_(probe,gallery, cosine_sim)
# dec = main(r)
# print(dec)
# for x in range(3,50, 2):
#     for y in range(1, 50):
r = thres_local(probe, gallery, hamming_, 29, 27)
dec = main(r)
# print(x, y)
print(dec)

r = adaptive_thres(probe, gallery, hamming_, 29, 27)
dec = main(r)
print(dec)

# print
# r = lbp(probe,gallery, hamming_, 15, 2, None)
# dec = main(r)
# print(dec)
