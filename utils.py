import cv2
import mahotas
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import hamming
from skimage import metrics
from skimage.feature import local_binary_pattern
from skimage.filters import threshold_otsu
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity


def preprocess(img):
    equ = cv2.equalizeHist(img)
    img = np.hstack((img, equ))
    img = saliency_map(img)
    return img


def saliency_map(img):
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    threshMap = cv2.threshold(saliencyMap, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    return threshMap


def create_sim_matrix(part):
    sim = []
    for i in range(1, 101):
        row = []
        for j in range(1, 101):
            gallery = galary_im(i)
            probe = probe_ims(j)
            if (part == 'I'):
                gallery = binarization(gallery, 128)
                probe = binarization(probe, 128)
            elif (part == 'II'):
                gallery = preprocess(gallery)
                probe = preprocess(probe)
            row.append(hamming_sim(gallery.flatten(), probe.flatten()))
        sim.append(row)
    return np.asarray(sim)


def d_index(genuine, imposter):
    mu_gen = np.mean(genuine)
    mu_imp = np.mean(imposter)
    std_gen = np.std(genuine)
    std_imp = np.std(imposter)
    return 2 ** 0.5 * (abs(mu_gen - mu_imp)) / (std_gen ** 2 + std_imp ** 2) ** 0.5


def gen_imp_score(sim):
    gen = np.diag(sim)
    diag_matrix = np.eye(sim.shape[0], dtype=bool)
    imp = sim[~diag_matrix]
    return gen, imp


def low_pass_filter(img, size):
    kernel = np.ones(size, np.float32) / (size[0] * size[1])
    return cv2.filter2D(img, -1, kernel)


def gaussian_filter(img):
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]]) / 16
    return cv2.filter2D(img, -1, kernel)


def high_pass_filter(img):
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

    return cv2.filter2D(img, -1, kernel)


def dog(img):
    gamma = 1
    img = np.power(img, gamma)

    # DOG
    blur1 = cv2.GaussianBlur(img, (0, 0), 1, borderType=cv2.BORDER_REPLICATE)
    blur2 = cv2.GaussianBlur(img, (0, 0), 2, borderType=cv2.BORDER_REPLICATE)
    img_dog = (blur1 - blur2)
    # normalize by the largest absolute value so range is -1 to
    img_dog = img_dog / np.amax(np.abs(img_dog))
    img_dog2 = (255.0 * (0.5 * img_dog + 0.5)).clip(0, 255).astype(np.uint8)
    return img_dog2


def binarization(img, thres):
    ret, bin_thres = cv2.threshold(img, thres, 1, cv2.THRESH_BINARY)
    return bin_thres


def binarization_using_cv2(img, type):
    th_ = cv2.adaptiveThreshold(img, 1, type, cv2.THRESH_BINARY, 31, 10)
    return th_


# uses hamming distance as similarity measure
def jaccard_sim(img1, img2):
    return jaccard_score(img1, img2, average='weighted')


def hamming_sim(img1, img2):
    return hamming(img1, img2)


def euc_sim(img1, img2):
    return np.linalg.norm(img1 - img2)


def cosine_sim(img1, img2):
    return cosine_similarity(img1, img2)


def pgm2numpy(filename):
    with open(filename, 'rb') as pgmf:
        im = plt.imread(pgmf)
    return im


def galary_im(subject):
    path = f'../GallerySet/subject{subject}_img1.pgm'
    return pgm2numpy(path)


def probe_ims(subject):
    path = f'../ProbeSet/subject{subject}_img2.pgm'
    return pgm2numpy(path)


def contrast_stretch(img):
    # Map the intensity values between the thresholds to the full range of pixel values
    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    img = cv2.LUT(img, table)
    return img

def lbp_(img, x, y, bins):
    lbph = local_binary_pattern(img, x, y).astype(np.uint8)
    n_bins = bins
    # print(lbph)
    hist, _ = np.histogram(lbph.flatten(), bins=n_bins, range=(0, n_bins - 1))
    lbph_eq = cv2.equalizeHist(lbph)
    hist = cv2.calcHist([lbph_eq], [0], None, [n_bins], [0, n_bins])

    hist = hist.astype('float32')
    hist /= (lbph.shape[0] * lbph.shape[1])
    lbph_hist = np.where(hist > np.mean(hist), 1, 0).astype('uint8')
    return lbph_hist


def chow_kaneko(image, max_iterations=1000, convergence_threshold=0.01):
    threshold = threshold_otsu(image)
    for i in range(max_iterations):
        foreground = image[image > threshold]
        background = image[image <= threshold]
        new_threshold = 0.5 * (np.mean(foreground) + np.mean(background))
        if abs(threshold - new_threshold) < convergence_threshold:
            break
        threshold = new_threshold
    # print(threshold)
    binarized = (image > threshold).astype('uint8')
    return binarized


def ssim(img1: np.ndarray, img2: np.ndarray):
    img1 = img1.flatten()
    img2 = img2.flatten()
    return metrics.structural_similarity(img1, img2)


def ridler_calvard(photo):
    # photo = utils.edge(photo)
    photo = photo.astype(np.uint8)

    T_rc = mahotas.rc(photo)
    img = photo > T_rc
    return img.astype('uint8')


def adaptive_(img, x, y):
    img = cv2.bilateralFilter(img, 5, 70, 70)
    img = contrast_stretch(img)
    img = contrast_stretch(img)
    img = contrast_stretch(img)

    thres_ = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, x, y)
    return thres_
