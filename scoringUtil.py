import skimage.io
import numpy as np
import skimage.segmentation
import csv


def get_image_size(id, data_path):
    image = skimage.io.imread(data_path + "{}/images/{}.png".format(id,id) )
    return image.shape


def get_feature_mask(output_str, image_size):
    h = image_size[0]
    output_list = output_str.split()
    feature_mask = np.zeros(image_size[:2], np.uint16)
    for i in range(0, len(output_list), 2):
        top_pixel = int(output_list[i]) - 1
        num_pixels = int(output_list[i + 1])
        top_pixel_c = top_pixel // h
        top_pixel_r = top_pixel % h
        for j in range(num_pixels):
            feature_mask[top_pixel_r + j][top_pixel_c] = 1
    return feature_mask

def get_rect_coords(output_str, image_size):
    h = image_size[0]
    output_list = output_str.split()
    output_list = np.array(output_list, np.uint16)
    top_pixels = output_list[::2] - 1
    bottom_pixels = top_pixels + output_list[1::2] - 1
    left_col = top_pixels[0] // h
    right_col = top_pixels[-1] // h
    top_pixels_r = top_pixels % h
    top_row = min(top_pixels_r)
    bottom_pixels_r = bottom_pixels % h
    bottom_row = max(bottom_pixels_r)
    return (top_row, left_col, bottom_row, right_col)

def do_rects_intersect(rect_1, rect_2):
    intersect = False
    (x1, y1, w1, z1) = rect_1
    (x2, y2, w2, z2) = rect_2
    if x1 <= w2 and y1 <= z2 and w1 >= x2 and z1 >= y2:
        intersect = True
    return intersect

def test_do_rects_intersect():
    rect1 = (1, 1, 5, 6)
    rect2 = (3, 3, 7, 6)
    rect3 = (10, 1, 12, 2)
    rect4 = (1, 10, 1, 10)
    print(do_rects_intersect(rect1, rect2))
    print(do_rects_intersect(rect2, rect1))
    print(do_rects_intersect(rect1, rect3))
    print(do_rects_intersect(rect3, rect1))
    print(do_rects_intersect(rect1, rect4))
    print(do_rects_intersect(rect4, rect1))


def get_image_prediction(features, image_size):
    image = np.zeros(image_size[:2], np.uint16)
    for i,feature_str in enumerate(features):
        feature_mask = get_feature_mask(feature_str, image_size)
        image[feature_mask > 0] = i+1
    return image

def get_rect_intersections(image_data, key_data, image_size):
    image_rect_list = [None]*len(image_data)
    key_rect_list = [None]*len(key_data)
    for i,feature in enumerate(image_data):
        image_rect_list[i] = get_rect_coords(feature, image_size) + (i + 1,)
    for i,feature in enumerate(key_data):
        key_rect_list[i] = get_rect_coords(feature, image_size) + (i + 1,)
    image_rect_list.sort()
    key_rect_list.sort()

    intersecting_pairs = []
    min_j = 0
    i = 0
    while i < len(image_rect_list):
        image_rect = image_rect_list[i]
        while min_j < len(key_rect_list) and image_rect[0] > key_rect_list[min_j][2]:
            min_j += 1
        j = min_j
        while j < len(key_rect_list) and image_rect[2] > key_rect_list[j][0]:
            if do_rects_intersect(image_rect[:4], key_rect_list[j][:4]):
                intersecting_pairs.append((i,j))
            j += 1
        i += 1

    pairs = [None]*len(intersecting_pairs)
    for j,(image_i, key_i) in enumerate(intersecting_pairs):
        pairs[j] = (image_rect_list[image_i][4], key_rect_list[key_i][4])
    return pairs

# Precision helper function
def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn


def get_score(image_data, key_data, image_size):
    y_pred = get_image_prediction(image_data, image_size)
    labels = get_image_prediction(key_data, image_size)

    # Compute number of objects
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))
    print("Number of true objects:", true_objects)
    print("Number of predicted objects:", pred_objects)

    '''
    # Compute intersection between all objects
    rect_intersections = get_rect_intersections(image_data, key_data, image_size)
    intersection = np.zeros([len(key_data) + 1, len(image_data) + 1])
    for (pred_feature_id, true_feature_id) in rect_intersections:
        pred_mask = np.zeros_like(y_pred)
        pred_mask[y_pred == pred_feature_id] = pred_feature_id
        true_mask = np.zeros_like(labels)
        true_mask[labels == true_feature_id] = true_feature_id
        temp = np.histogram2d(true_mask.flatten(), pred_mask.flatten(), bins=(2, 2))[0][1][1]
        intersection[true_feature_id, pred_feature_id] = temp



    for i in range(len(key_data)):
        mask = np.zeros_like(labels)
        mask[labels == i+1] = 1
        intersection[i + 1][0] = np.sum(mask, axis=(0, 1)) - np.sum(intersection, axis = 1)[i + 1]
    for i in range(len(image_data)):
        mask = np.zeros_like(y_pred)
        mask[y_pred == i+1] = 1
        intersection[0][i + 1] = np.sum(mask, axis=(0, 1)) - np.sum(intersection, axis = 0)[i + 1]

    intersection[0][0] = image_size[0] * image_size[1] - np.sum(intersection)
    '''
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

def get_image_dict(csv_filename):
    image_dict = {}
    with open(csv_filename) as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader, None)
        for row in csv_reader:
            id = row[0]
            feature_str = row[1]
            if id in image_dict:
                image_dict[id].append(feature_str)
            else:
                image_dict[id] = [feature_str]
    return image_dict

def score(output_csv, key_csv, data_path):

    image_dict = get_image_dict(output_csv)
    key_dict = get_image_dict(key_csv)

    image_list = list(image_dict.keys())
    image_stats_dict = dict.fromkeys(image_list)
    image_score_dict = dict.fromkeys(image_list)
    for i,image_id in enumerate(image_list):
        image_size = get_image_size(image_id, data_path)
        [image_score, image_stats] = get_score(image_dict[image_id], key_dict[image_id], image_size)
        #[image_score, image_stats] = get_score(labels, pred)
        image_stats_dict[image_id] = image_stats
        image_score_dict[image_id] = image_score
        print(i)

    score_list = list(image_score_dict.values())
    return [sum(score_list)/len(score_list), image_score_dict, image_stats_dict]