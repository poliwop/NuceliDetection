import numpy as np
from PIL import Image
from os import listdir
import pickle

training_data_path = 'data/stage1_train/'
pickle_data_path = ''
pickle_file_name = 'image_matrices.p'

image_list = [0]*len(listdir(training_data_path))

print "converting images to matrices"
for i in range(len(listdir(training_data_path))):
    folder = listdir(training_data_path)[i]
    img_path = training_data_path + folder + '/images/' + folder + '.png'
    img = Image.open(img_path).convert('L')
    image_list[i] = np.array(img)

print "pickling"
pickle.dump(image_list, open(pickle_data_path + pickle_file_name, "wb"))