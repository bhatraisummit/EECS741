from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
import cv2
import os

# img_path = '/Users/summit/PycharmProjects/pythonProject/EECS741_CVProject/GallerySet/subject1_img1.pgm'
# img = Image.open(img_path)


def imgtoRGB(img):
    dct_R = img
    dct_G = img
    dct_B = img
    return np.dstack([dct_R, dct_G, dct_B])


# Create an inception resnet (in eval mode):
def facenet_embeddings(img):
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    img = imgtoRGB(img)

    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)

    transform = transforms.Compose([transforms.ToTensor()])

    img = transform(img)
    img_embedding = resnet(img.unsqueeze(0))
    return img_embedding.detach().numpy()


# print(facenet_embeddings(img).shape)
