import skimage.measure
import numpy as np

#   Applies a threshold and labels connected components of the resulting image, for each image in list
def run_basic_predict(image_list, params):
    # image_list    List of pairs [image_id, image_mat] where image_mat is an ndarray representing a grayscale image
    # threshold     number in (0, 256)
    #
    # returns       List of pairs [image_id, labeled_im] where labeled_im is an ndarray with labeled connected
    #               components

    labeled_list = [None]*len(image_list)
    for i,image in enumerate(image_list):
        if i%50 == 0:
            print(str(i) + ' out of ' + str(len(image_list)))
        classified_image = process_image(image[1], params)
        labeled_image = skimage.measure.label(classified_image)
        labeled_list[i] = [image[0], labeled_image]

    return labeled_list


def process_image(image, params):
    # Invert if light background
    [threshold, intensity_threshold, q] = params
    if image.mean() / 255 >= intensity_threshold:
        image = 255 - image

    [bot_val, top_val] = np.percentile(image, [0, 100-q])
    image -= bot_val
    image *= (255.0 / top_val)
    image[image < 0] = 0
    image[image > 255] = 255

    classified_image = np.zeros_like(image)
    classified_image[image > threshold] = 1
    return classified_image

