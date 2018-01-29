from nucleiUtil import get_image_list
import csv
import numpy as np

def lower_resolution(array, factor):
    intensity_threshold = 0.3
    input_columns = array.shape[1]
    input_rows = array.shape[0]
    output_columns = input_columns//factor
    output_rows = input_rows//factor
    output = np.zeros(shape=(output_rows, output_columns))
    for i in range(output_rows):
        for j in range(output_columns):
            pixel = 0
            for k in range(factor):
                for l in range(factor):
                    pixel += array[i*factor+k][j*factor+l]
            output[i][j]=float(pixel)/(factor*factor)
    if np.average(output)/255 >= intensity_threshold:
        output = 255-output
    output *= (255.0/output.max())
    return output

def recursive_label(im, lab_im, label, i, j):
    print(i)
    print(j)
    print(label)
    print('ah')
    lab_im[i,j] = label
    if i+1 < lab_im.shape[0] and lab_im[i+1,j] == 0 and im[i+1,j] == 1:
        recursive_label(im, lab_im, label, i+1, j)
    if i > 0 and lab_im[i-1,j] == 0 and im[i-1,j] == 1:
        recursive_label(im, lab_im, label, i-1, j)
    if j+1 < lab_im.shape[1] and lab_im[i,j+1] == 0 and im[i,j+1] == 1:
        recursive_label(im, lab_im, label, i, j+1)
    if j > 0 and lab_im[i,j-1] == 0 and im[i,j-1] == 1:
        recursive_label(im, lab_im, label, i, j-1)

def image_labeller(im):
    labelled_image = np.zeros_like(im)
    label = 1
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i,j] == 1 and labelled_image[i,j] == 0:
                recursive_label(im, labelled_image, label, i, j)
                label += 1
    return labelled_image

def label_image(image, threshold):

    # Invert if light background
    intensity_threshold = 0.3
    if np.average(image)/255 >= intensity_threshold:
        image = 255 - image
    image *= (255.0/image.max())
    classified_image = np.zeros_like(image)
    classified_image[image > threshold] = 1

    labeled_image = image_labeller(classified_image)
    print('pause')

def run_basic_predict(output_filename, image_list, threshold):

    # Get labeled image
    labeled_list = [None]*len(image_list)
    for i,image in enumerate(image_list):
        labeled_list[i] = label_image(image[1], threshold)


    # Convert to output format and write to file

    '''
    with open(output_filename, 'w') as outputfile:
        writer = csv.writer(outputfile, lineterminator='\n')
        writer.writerow(['ImageId', 'EncodedPixels'])
        for counter,image in enumerate(image_list):
            test = image[1].shape[0]*image[1].shape[1]
            height =image[1].shape[0] #height of original image in pixels
            pixelated = lower_resolution(image[1], factor)
            if pixelated.max() == 0:
                writer.writerow([image[0], '1 1'])
            for i in range(pixelated.shape[0]):
                for j in range(pixelated.shape[1]):
                    if pixelated[i][j] > threshold:
                        pixels = ''
                        for k in range(factor):
                            top_pixel = (j*factor + k)*height + i*factor + 1
                            pixels += str(top_pixel)+' ' + str(factor) + ' '
                            if top_pixel > test:
                                print(image[0])
                                print(test)
                                print(top_pixel)
                                print([i, j, pixelated[i][j]])
                                raise
                        writer.writerow([image[0], pixels])
            if counter % 10 == 0:
                print('starting image number '+str(counter))
    '''

testing_data_path = 'data/stage1_train/'
output_filename = 'prediction.csv'
image_list = get_image_list(testing_data_path)
threshold = 120 #keep pixels at least this bright
run_basic_predict(output_filename, image_list, threshold)