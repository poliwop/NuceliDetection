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

def run_lowrespredict(output_filename, image_list, factor, threshold):
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


testing_data_path = 'data/stage1_test/'
output_filename = 'prediction.csv'
image_list = get_image_list(testing_data_path)
threshold = 120 #keep pixels at least this bright
factor = 14 #compress width/height by factor of 16
run_lowrespredict(output_filename, image_list, factor, threshold)