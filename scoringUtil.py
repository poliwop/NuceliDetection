import skimage.io
import numpy as np
import skimage.segmentation
import csv


def get_image_size(id, data_path):
    image = skimage.io.imread(data_path + "{}/images/{}.png".format(id,id) )
    return image.shape


def get_feature_mask(output_str, image_size):
    h = image_size[0]
    w = image_size[1]
    output_list = output_str.split()
    feature_mask = np.asarray([[0]*w]*h, np.uint16)
    for i in range(0, len(output_list), 2):
        top_pixel = int(output_list[i]) - 1
        num_pixels = int(output_list[i + 1])
        top_pixel_c = top_pixel // h
        top_pixel_r = top_pixel % h
        for j in range(num_pixels):
            feature_mask[top_pixel_r + j][top_pixel_c] = 1
    return feature_mask

def get_labels(id, data_path):
    # Load a single image and its associated masks
    file = data_path + "{}/images/{}.png".format(id,id)
    masks = data_path + "{}/masks/*.png".format(id)
    image = skimage.io.imread(file)

    masks = skimage.io.imread_collection(masks).concatenate()
    height, width, _ = image.shape
    num_masks = masks.shape[0]

    # Make a ground truth label image (pixel value is index of object label)
    labels = np.zeros((height, width), np.uint16)
    for index in range(0, num_masks):
        labels[masks[index] > 0] = index + 1
    return labels

def get_image_prediction(features, image_size):
    image = np.zeros(image_size[:2], np.uint16)
    for i,feature_str in enumerate(features):
        feature_mask = get_feature_mask(feature_str, image_size)
        image += (i+1)*feature_mask
    return image


# Precision helper function
def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn


def get_score(labels, y_pred):

    # Compute number of objects
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))
    print("Number of true objects:", true_objects)
    print("Number of predicted objects:", pred_objects)

    # Compute intersection between all objects
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Loop over IoU thresholds
    prec = []
    image_stats = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        p = tp*1. / (tp + fp + fn)
        image_stats.append([t, tp, fp, fn, p])
        prec.append(p)
    return [np.mean(prec), image_stats]

def score(output_csv, data_path):

    image_dict = {}
    with open(output_csv) as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader, None)
        for row in csv_reader:
            id = row[0]
            feature_str = row[1]
            if id in image_dict:
                image_dict[id].append(feature_str)
            else:
                image_dict[id] = [feature_str]

    image_list = list(image_dict.keys())
    image_stats_dict = dict.fromkeys(image_list)
    image_score_dict = dict.fromkeys(image_list)
    for i,image_id in enumerate(image_list):
        image_size = get_image_size(image_id, data_path)
        pred = get_image_prediction(image_dict[image_id], image_size)
        labels = get_labels(image_id, data_path)
        [image_score, image_stats] = get_score(labels, pred)
        image_stats_dict[image_id] = image_stats
        image_score_dict[image_id] = image_score
        print(i)

    score_list = list(image_score_dict.values())
    return [sum(score_list)/len(score_list), image_score_dict, image_stats_dict]