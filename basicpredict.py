import skimage.measure
import numpy as np

#   Applies a threshold and labels connected components of the resulting image, for each image in list
def run_basic_predict(image_list, threshold):
    # image_list    List of pairs [image_id, image_mat] where image_mat is an ndarray representing a grayscale image
    # threshold     number in (0, 256)
    #
    # returns       List of pairs [image_id, labeled_im] where labeled_im is an ndarray with labeled connected
    #               components

    labeled_list = [None]*len(image_list)
    for i,image in enumerate(image_list):
        if i%50 == 0:
            print(str(i) + ' out of ' + str(len(image_list)))
        classified_image = process_image(image[1], threshold)
        labeled_image = skimage.measure.label(classified_image)
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

