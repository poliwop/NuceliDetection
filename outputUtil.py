import numpy as np
import csv

def write_output_file(labeled_list, filename):
    # labeled_list  List of pairs [image_id, labeled_im] where labeled_im is an ndarray with labeled connected
    #               components
    # filename      Full path to output file

    rows = []
    for i, labeled_image in enumerate(labeled_list):
        if i%10 == 0:
            print('writing ' + str(i) + ' out of ' + str(len(labeled_list)))
        feature_list = get_feature_list(labeled_image[1])
        for feature_str in feature_list:
            rows.append([labeled_image[0], feature_str])

    with open(filename, 'w') as output_file:
        writer = csv.writer(output_file, lineterminator='\n')
        writer.writerow(['ImageId', 'EncodedPixels'])
        writer.writerows(rows)

def get_feature_list(im):
    feature_list = []
    max_feature = int(im.max())
    for i in range(max_feature):
        feature = np.zeros_like(im)
        feature[im == i+1] = 1
        feature_str = get_feature_str(feature)
        if len(feature_str) > 0:
            feature_list.append(feature_str)
    return feature_list

def get_feature_str(im):
    h = im.shape[0]
    w = im.shape[1]
    pairs = []
    for i in range(w):
        col = im[:,i]
        [starts, heights] = get_feature_starts(col)
        pairs.extend(zip(i*h + starts + 1, heights))
    out_list = [str(x) for pair in pairs for x in pair]
    out_str = " ".join(out_list)
    return out_str

def get_feature_starts(col):
    col_diff = np.empty_like(col)
    col_diff[0] = col[0]
    col_diff[1:] = (col[1:] - col[:-1])
    feature_starts = np.where(col_diff == 1)[0]
    feature_ends = np.where(col_diff == -1)[0]
    if len(feature_ends) < len(feature_starts):
        feature_ends = np.append(feature_ends, len(col))
    feature_heights = feature_ends - feature_starts
    return [feature_starts, feature_heights]