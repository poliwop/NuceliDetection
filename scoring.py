
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import skimage.segmentation
import csv

training_data_path = 'data/stage1_train/'

# Load a single image and its associated masks
id = '0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9'
file = training_data_path + "{}/images/{}.png".format(id,id)
masks = training_data_path + "{}/masks/*.png".format(id)
image = skimage.io.imread(file)


def get_image_size(id, data_path):
    image = skimage.io.imread(data_path + "{}/images/{}.png".format(id,id) )
    return image.shape

def get_pixel_list(output_str):
    output_list = output_str.split()
    pixel_list = []
    for i in range(0, len(output_list), 2):
        top_pixel = int(output_list[i])
        num_pixels = int(output_list[i + 1])
        pixel_list.extend(range(top_pixel, top_pixel + num_pixels))
    return pixel_list

def get_feature_image(output_str, image_size):
    h = image_size[0]
    w = image_size[1]
    output_list = output_str.split()
    feature_image = [[0]*w]*h
    for i in range(0, len(output_list), 2):
        top_pixel = int(output_list[i])
        num_pixels = int(output_list[i + 1])
        top_pixel_c = top_pixel / h
        top_pixel_r = top_pixel % h
        for j in range(num_pixels):
            feature_image[top_pixel_r + j][top_pixel_c] = (i/2) + 1
    return feature_image





def get_image_dict(output_csv):
    image_dict = {}
    with open(output_csv, 'rb') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)
        for row in csvreader:
            image_id = row[0]
            #image_size = get_image_size(image_id, data_path)
            pixel_list = get_pixel_list(row[1])
            if image_dict.has_key(image_id):
                image_dict[image_id].append(pixel_list)
            else:
                image_dict[image_id] = [pixel_list]
    return image_dict





masks = skimage.io.imread_collection(masks).concatenate()
height, width, _ = image.shape
num_masks = masks.shape[0]

# Make a ground truth label image (pixel value is index of object label)
labels = np.zeros((height, width), np.uint16)
for index in range(0, num_masks):
    labels[masks[index] > 0] = index + 1

# Show label image
#fig = plt.figure()
#plt.imshow(image)
#plt.title("Original image")
#fig = plt.figure()
#plt.imshow(labels)
#plt.title("Ground truth masks")


def get_image_score(pixels_1, pixels_2):
    overlap_matrix = [[0]*len(pixels_1)]*len(pixels_2)
    pixels_1.sort()
    pixels_2.sort()
    for feature in pixels_1:
        f_min = feature[0]
        f_max = feature[-1]

image_dict = get_image_dict('predictionTrain.csv')
solution_dict = get_image_dict('data/stage1_train_labels.csv')
image_score = []
#for key in image_dict:
#    image_score.append(get_image_score(image_dict[key], solution_dict[key]))

# Simulate an imperfect submission
offset = 2 # offset pixels
y_pred = labels[offset:, offset:]
y_pred = np.pad(y_pred, ((0, offset), (0, offset)), mode="constant")
y_pred[y_pred == 20] = 0 # Remove one object
y_pred, _, _ = skimage.segmentation.relabel_sequential(y_pred) # Relabel objects

# Show simulated predictions
fig = plt.figure()
plt.imshow(y_pred)
plt.title("Simulated imperfect submission")

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

# Precision helper function
def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn

# Loop over IoU thresholds
prec = []
print("Thresh\tTP\tFP\tFN\tPrec.")
for t in np.arange(0.5, 1.0, 0.05):
    tp, fp, fn = precision_at(t, iou)
    p = tp*1. / (tp + fp + fn)
    print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
    prec.append(p)
print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))

