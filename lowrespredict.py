from nucleiUtil import get_image_list
import csv
import numpy as np

testing_data_path = 'data folder/stage1_test/'
image_list = get_image_list(testing_data_path)
threshhold = 200 #keep pixels at least this bright
factor = 16 #compress width/height by factor of 16

def lower_resolution(array, factor):
    input_columns = array.shape[1]
    input_rows = array.shape[0]
    output_columns = input_columns/factor
    output_rows = input_rows/factor
    output = np.zeros(shape=(output_rows, output_columns))
    if (input_rows % 16) or (input_columns % 16):
        return output
    i=0
    while i < output_rows:
        j=0
        while j < output_columns:
            pixel = 0
            k=0
            while k < factor:
                l=0
                while l < factor:
                    pixel += array[i*factor+k][j*factor+l]
                    l += 1
                k += 1
            output[i][j]=float(pixel)/(factor*factor)
            j += 1
        i += 1
    output *= (255.0/output.max())
    return output

with open('prediction.csv', 'w') as outputfile:
    writer = csv.writer(outputfile, lineterminator='\n')
    writer.writerow(['ImageId', 'EncodedPixels'])
    counter=0
    for image in image_list:
        test = image[1].shape[0]*image[1].shape[1]
        height =image[1].shape[0] #height of original image in pixels
        pixelated = lower_resolution(image[1], factor)
        i=0
        if pixelated.max() == 0:
            writer.writerow([image[0], '1 1'])
        while i < pixelated.shape[0]:
            j=0
            while j < pixelated.shape[1]:
                if pixelated[i][j] > threshhold:
                    pixels = ''
                    for k in range(factor):
                        pixels += str(j*16*height+i*factor+k*height+1)+' 16 '
                        if j*16*height+i*factor+k*height+1 > test:
                            print image[0]
                            print test
                            print j*16*height+i*factor+k*height+1
                            print i, j, pixelated[i][j]
                            raise
                    writer.writerow([image[0], pixels])
                j+=1
            i+=1
        counter += 1
        if counter % 10 == 0:
            print 'starting image number'+str(counter)
