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
        feature_ls = [str(x) for x in rle_encoding(feature)]
        feature_str = " ".join(feature_ls)
        if len(feature_str) > 0:
            feature_list.append(feature_str)
    return feature_list


# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths