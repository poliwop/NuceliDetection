from nucleiUtil import get_image_list
import pickle

training_data_path = 'data/stage1_train/'
pickle_data_path = ''
pickle_file_name = 'image_matrices.p'

image_list = get_image_list(training_data_path)

print "pickling"
pickle.dump(image_list, open(pickle_data_path + pickle_file_name, "wb"))