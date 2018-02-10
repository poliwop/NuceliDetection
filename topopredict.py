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
        if i%50 == 0:
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
    component_cts = np.empty(256)
    for t in range(0,256,1):
        classified_image = np.zeros_like(image)
        classified_image[image > t] = 1
        [_, component_ct] = skimage.measure.label(classified_image, return_num=True)
        component_cts[t] = component_ct
        #feature_props_list = skimage.measure.regionprops(labeled_image)
        #feature_areas = []
        #for feature in feature_props_list:
        #    if feature.area > min_area and feature.area < max_area:
        #        feature_areas.append(feature.area)
        #areas.extend(feature_areas)
        #areas_by_t.append([t, feature_areas])

    #fig = plt.gcf()
    peak_i = component_cts.argmax()
    peak2_i = component_cts[peak_i+15:].argmax() + peak_i + 15
    dip_i = component_cts[peak_i:peak2_i].argmin() + peak_i
    threshold = (peak2_i + dip_i)/2

    #plt.plot(component_cts[dip_i:])
    #plt.show()

    #classified_image = np.zeros_like(image)
    #classified_image[image > threshold] = 1
    #labeled_image = skimage.measure.label(classified_image)

    #fig = plt.figure(figsize=(image.shape[0], image.shape[1]))
    #fig.add_subplot(2, 1, 1)
    #plt.imshow(image)
    #fig.add_subplot(2, 1, 2)
    #plt.imshow(labeled_image)
    #plt.show()
    print(threshold)

    return threshold
