from nucleiUtil import get_image_list, set_min_area
from outputUtil import write_output_file
from scoringUtil import score_matrices
import numpy as np
import sys
import cProfile

testing_data_path = 'data/stage1_test/'
output_filename = 'prediction.csv'
key_csv = 'data/stage1_train_labels.csv'
threshold = 50 #keep pixels at least thi# s bright
min_area = 10

def run_basic_predict(image_list, threshold):

    # Get labeled image
    labeled_list = [None]*len(image_list)
    feature_ct = [None]*len(image_list)
    for i,image in enumerate(image_list):
        if i%10 == 0:
            print(str(i) + ' out of ' + str(len(image_list)))
        classified_image = process_image(image[1], threshold)
        labeled_image = image_labeler(classified_image)
        labeled_list[i] = [image[0], labeled_image]

    return labeled_list


def process_image(image, threshold):
    # Invert if light background
    image -= image.min()
    image *= (255.0 / image.max())
    intensity_threshold = 0.5
    if np.average(image) / 255 >= intensity_threshold:
        image = 255 - image
    classified_image = np.zeros_like(image)
    classified_image[image > threshold] = 1
    return classified_image


def image_labeler(im):
    labeled_image = np.zeros_like(im)
    label = 0
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i,j] == 1 and labeled_image[i,j] == 0:
                label += 1
                label_component(im, labeled_image, label, i, j)

    return labeled_image

def label_component(im, lab_im, label, i, j):
    stack = [[-1,-1]]
    lab_im[i,j] = label
    while stack:
        if i + 1 < lab_im.shape[0] and im[i + 1, j] == 1 and lab_im[i + 1, j] == 0:
            lab_im[i+1,j] = label
            stack.append([i+1,j])
        if i > 0 and im[i - 1, j] == 1 and lab_im[i - 1, j] == 0:
            lab_im[i-1, j] = label
            stack.append([i-1, j])
        if j + 1 < lab_im.shape[1] and im[i, j + 1] == 1 and lab_im[i, j + 1] == 0:
            lab_im[i, j+1] = label
            stack.append([i, j+1])
        if j > 0 and im[i, j - 1] == 1 and lab_im[i, j - 1] == 0:
            lab_im[i, j-1] = label
            stack.append([i, j-1])
        [i,j] = stack.pop()



image_list = get_image_list(testing_data_path)
print("getting predictions")
labeled_list = run_basic_predict(image_list, threshold)

print("removing tiny features")
set_min_area(labeled_list, min_area)
#print("scoring")
#myscore = score_matrices(labeled_list, key_csv)
#print(myscore[0])
print("writing output")
write_output_file(labeled_list, output_filename)


#cProfile.run('temp_function()')
