import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
import cv2
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import facenet_embeddings


def preprocess (img):
    # img = low_pass_filter(img, (5,5))
    # img = contrast_stretch(img)
    # img = cv2.medianBlur(img, 3)
    img = saliency_map(img)
    # img = local_binary_pattern(img, 24, 3, method='uniform')
    # img = dog(img)
    # print(img.max(), img.min())
    return img

def saliency_map (img):
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    # Initialise the more fine-grained saliency detector and compute the saliencyMap
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    threshMap = cv2.threshold(saliencyMap, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Show Images
    # plt.imshow(threshMap, cmap='gray')
    # plt.imshow(saliencyMap, cmap='gray')
    # plt.imshow(img)
    # plt.show()
    return threshMap

# img = cv2.imread('../GallerySet/subject2_img1.pgm', cv2.IMREAD_GRAYSCALE)
# saliency_map(img)

def low_pass_filter (img, size) :
    kernel = np.ones(size, np.float32) / (size[0]*size[1])
    return cv2.filter2D(img, -1, kernel)

def gaussian_filter (img):
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]]) / 16
    return cv2.filter2D(img, -1, kernel)

def high_pass_filter (img):
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
    # kernel = np.array([[-1, -1, -1, -1, -1],
    #                    [-1, 1, 2, 1, -1],
    #                    [-1, 2, 4, 2, -1],
    #                    [-1, 1, 2, 1, -1],
    #                    [-1, -1, -1, -1, -1]])

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
    # im1_norm = np.linalg.norm(img1)
    # im2_norm = np.linalg.norm(img2)
    # im_1_2dot = img1 @ img2
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


def create_sim_matrix(binarization_type):
    sim = []
    for i in range(1, 101):
        row = []
        for j in range(1, 101):
            gallery = galary_im(i)
            probe = probe_ims(j)
            gallery = preprocess(gallery)
            probe = preprocess(probe)
            # if (binarization_type == 'bin'):
            #     g = binarization(gallery, 128)
            #     p = binarization(probe, 128)

            # elif (binarization_type == cv2.ADAPTIVE_THRESH_MEAN_C):
            #     g = binarization_using_cv2(gallery, cv2.ADAPTIVE_THRESH_MEAN_C).flatten()
            #     p = binarization_using_cv2(probe, cv2.ADAPTIVE_THRESH_MEAN_C).flatten()
            #
            # elif (binarization_type == cv2.ADAPTIVE_THRESH_GAUSSIAN_C):
            #     g = binarization_using_cv2(gallery, cv2.ADAPTIVE_THRESH_GAUSSIAN_C).flatten()
            #     p = binarization_using_cv2(probe, cv2.ADAPTIVE_THRESH_GAUSSIAN_C).flatten()
            # g = facenet_embeddings(g).flatten()
            # p = facenet_embeddings(p).flatten()
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


def contrast_stretch(img):
    # Map the intensity values between the thresholds to the full range of pixel values
    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    img = cv2.LUT(img, table)
    return img
