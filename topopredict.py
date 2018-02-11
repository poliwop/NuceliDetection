import skimage.measure
import numpy as np
import matplotlib.pyplot as plt

#   Applies a threshold and labels connected components of the resulting image, for each image in list
def run_topo_predict(image_list, intensity_threshold):
    # image_list    List of pairs [image_id, image_mat] where image_mat is an ndarray representing a grayscale image
    # threshold     number in (0, 256)
    #
    # returns       List of pairs [image_id, labeled_im] where labeled_im is an ndarray with labeled connected
    #               components

    labeled_list = [None]*len(image_list)
    for i,image in enumerate(image_list):
        if i%10 == 0:
            print(str(i) + ' out of ' + str(len(image_list)))

        # label components
        print(image[0])
        labeled_image = process_image(image[1], intensity_threshold)
        # Add image name
        labeled_list[i] = [image[0], labeled_image]

    return labeled_list


def process_image(image, intensity_threshold):
    # Invert if light background
    #[threshold, intensity_threshold, _] = params
    processed_image = preprocess_image(image, intensity_threshold)
    threshold = get_threshold(processed_image)

    classified_image = np.zeros_like(processed_image)
    classified_image[processed_image > threshold] = 1
    labeled_image = skimage.measure.label(classified_image)

    return labeled_image

def preprocess_image(image, intensity_threshold):
    processed_image = np.copy(image)
    if image.mean() / 255 >= intensity_threshold:
        processed_image = 255 - processed_image
    return processed_image

    #[bot_val, top_val] = np.percentile(processed_image, [0, 100])
    #processed_image -= bot_val
    #processed_image *= (255.0 / top_val)
    #processed_image[processed_image < 0] = 0
    #processed_image[processed_image > 255] = 255

def get_threshold(image):

    #areas = []
    #areas_by_t = []
    #min_area = 10
    #max_area = image.shape[0]*image.shape[1]/2
    #max_area = 1000

    min_brightness = image.min().round().astype('uint8')
    max_brightness = image.max().round().astype('uint8')
    component_cts = np.empty(max_brightness + 1 - min_brightness, dtype='int32')
    for t in range(min_brightness,max_brightness + 1):
        classified_image = np.zeros_like(image)
        classified_image[image > t] = 1
        [_, component_ct] = skimage.measure.label(classified_image, return_num=True)
        component_cts[t - min_brightness] = component_ct
        #feature_props_list = skimage.measure.regionprops(labeled_image)
        #feature_areas = []
        #for feature in feature_props_list:
        #    if feature.area > min_area and feature.area < max_area:
        #        feature_areas.append(feature.area)
        #areas.extend(feature_areas)
        #areas_by_t.append([t, feature_areas])

    #fig = plt.gcf()

    peak_i = component_cts.argmax()
    shift = min(int((len(component_cts) - peak_i)/4), len(component_cts[peak_i:]) - 1)
    #shift = 15
    dip_i = component_cts[peak_i:peak_i+shift].argmin() + peak_i
    peak2_i = component_cts[dip_i:].argmax() + dip_i

    weight = 1/2
    threshold = weight*peak2_i + (1-weight)*dip_i + min_brightness

    print(threshold)

    plt.plot(component_cts)
    plt.show()

    classified_image = np.zeros_like(image)
    classified_image[image > threshold] = 1
    labeled_image = skimage.measure.label(classified_image)

    classified_image2 = np.zeros_like(image)
    classified_image2[image > 50] = 1
    labeled_image2 = skimage.measure.label(classified_image2)

    fig = plt.figure(figsize=(image.shape[0], image.shape[1]))
    fig.add_subplot(1, 3, 1)
    plt.imshow(image)
    fig.add_subplot(1, 3, 2)
    plt.imshow(labeled_image)
    fig.add_subplot(1, 3, 3)
    plt.imshow(labeled_image2)
    plt.show()


    return threshold
