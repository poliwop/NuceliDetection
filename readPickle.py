import pickle
pickle_data_path = ''
pickle_file_name = 'image_matrices.p'
test = pickle.load(open(pickle_data_path + pickle_file_name, "rb"))
print "hi"