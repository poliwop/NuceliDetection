from PIL import Image
from os import listdir
import numpy as np

def get_image_list(dir):
    image_list = [0]*len(listdir(dir))

    print "converting images to matrices"
    for i in range(len(listdir(dir))):
        folder = listdir(dir)[i]
        img_path = dir + folder + '/images/' + folder + '.png'
        img = Image.open(img_path).convert('L')
        image_list[i] = np.array(img)
    return image_list