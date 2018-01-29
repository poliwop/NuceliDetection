from PIL import Image
import skimage.io
import skimage.color
from os import listdir
import numpy as np

def get_image_list(dir):
    image_list = [None]*len(listdir(dir))

    print("converting images to matrices")
    for i in range(len(listdir(dir))):
        image_id = listdir(dir)[i]
        img_path = dir + image_id + '/images/' + image_id + '.png'
        img = skimage.io.imread(img_path)
        img = skimage.color.rgb2gray(img)*255
        image_list[i] = [image_id, img]
    return image_list