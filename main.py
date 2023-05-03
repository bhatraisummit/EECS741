from utils import *
from skimage import feature

if __name__ == '__main__':
    # PART I
    sim_mat = create_sim_matrix('bin')
    gen, imp = gen_imp_score(sim_mat)
    print(d_index(gen, imp))
    # #
    # #PART 2
    # sim_mat = create_sim_matrix(cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    # gen, imp = gen_imp_score(sim_mat)
    # print(d_index(gen, imp))
    #
    # sim_mat = create_sim_matrix(cv2.ADAPTIVE_THRESH_MEAN_C)
    # gen, imp = gen_imp_score(sim_mat)
    # print(d_index(gen, imp))

    # img = cv2.imread('../GallerySet/subject2_img1.pgm', cv2.IMREAD_GRAYSCALE)
    # # gamma correction
    # img_gamma = np.power(img, 0.2)
    # img_gamma2 = (255.0 * img_gamma).clip(0, 255).astype(np.uint8)
    #
    # # DOG
    # blur1 = cv2.GaussianBlur(img_gamma, (0, 0), 0.1, borderType=cv2.BORDER_REPLICATE)
    # blur2 = cv2.GaussianBlur(img_gamma, (0, 0), 0.2, borderType=cv2.BORDER_REPLICATE)
    # img_dog = (blur1 - blur2)
    # # normalize by the largest absolute value so range is -1 to
    # img_dog = img_dog / np.amax(np.abs(img_dog))
    # img_dog2 = (255.0 * (0.5 * img_dog + 0.5)).clip(0, 255).astype(np.uint8)
    #
    # ret, th1 = cv2.threshold(img_dog2, 120, 255, cv2.THRESH_BINARY)
    #
    # img2 = cv2.imread('../ProbeSet/subject2_img2.pgm', cv2.IMREAD_GRAYSCALE)
    # img_gamma = np.power(img2, 0.2)
    # img_gamma2 = (255.0 * img_gamma).clip(0, 255).astype(np.uint8)
    #
    # # DOG
    # blur1 = cv2.GaussianBlur(img_gamma, (0, 0), 0.1, borderType=cv2.BORDER_REPLICATE)
    # blur2 = cv2.GaussianBlur(img_gamma, (0, 0), 0.2, borderType=cv2.BORDER_REPLICATE)
    # img_dog = (blur1 - blur2)
    # # normalize by the largest absolute value so range is -1 to
    # img_dog = img_dog / np.amax(np.abs(img_dog))
    # img_dog2 = (255.0 * (0.5 * img_dog + 0.5)).clip(0, 255).astype(np.uint8)
    # ret, th2 = cv2.threshold(img_dog2, 120, 255, cv2.THRESH_BINARY)
    #
    # # img = cv2.GaussianBlur(img, (5, 5), 0)
    # # # img = cv2.Canny(img, 220, 250)
    # #
    # # img2 = cv2.GaussianBlur(img2, (5, 5), 0)
    # # # img2 = cv2.Canny(img2, 240, 250)
    #
    # # th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 29, 27)
    # th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 29, 27)
    # th4 = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 5)
    # th5 = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
    # titles = ['Original Image', 'Global Thresholding (v = 128)',
    #           'Adaptive Mean Th Gallery', 'Adaptive Gaussian Th Gallery',
    #           'Adaptive Mean Th Probe', 'Adaptive Gaussian Th Probe']
    # images = [img, img2, th1, th2, th4, th5]
    # for i in range(6):
    #     plt.subplot(3, 2, i + 1), plt.imshow(images[i], 'gray')
    #     plt.title(titles[i])
    #     plt.xticks([]), plt.yticks([])
    # plt.show()
