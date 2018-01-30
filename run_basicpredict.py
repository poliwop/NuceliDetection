from basicpredict import run_basic_predict
from nucleiUtil import *
from scoringUtil import score_matrices
from outputUtil import write_output_file
import cProfile

testing_data_path = 'data/stage1_train/'
output_filename = 'prediction.csv'
key_csv = 'data/stage1_train_labels.csv'
threshold = 50 #keep pixels at least thi# s bright
min_area = 10
min_areas = [1]

def temp_function():
    image_list = get_image_list(testing_data_path)
    print("getting predictions")
    labeled_list = run_basic_predict(image_list, threshold)
    results = []
    min_areas.sort()
    for a in min_areas:
        print("removing tiny features")
        set_min_area(labeled_list, a)
        print("scoring")
        myscore = score_matrices(labeled_list, key_csv)
        results.append((a,myscore[0]))
        print(results[-1])
    for r in results:
        print(r)
#print("writing output")
#write_output_file(labeled_list, output_filename)


cProfile.run('temp_function()')