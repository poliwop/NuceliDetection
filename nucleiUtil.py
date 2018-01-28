from PIL import Image
from os import listdir
import numpy as np

def get_image_list(dir):
    image_list = [None]*len(listdir(dir))

    print("converting images to matrices")
    for i in range(len(listdir(dir))):
        image_id = listdir(dir)[i]
        img_path = dir + image_id + '/images/' + image_id + '.png'
        img = Image.open(img_path).convert('L')
        image_list[i] = [image_id, np.array(img)]
    return image_list