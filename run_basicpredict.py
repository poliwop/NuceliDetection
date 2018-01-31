from basicpredict import run_basic_predict
from nucleiUtil import *
from scoringUtil import score_matrices
from outputUtil import write_output_file
import cProfile

testing_data_path = 'data/stage1_test/'
output_filename = 'prediction25-50-05.csv'
key_csv = 'data/stage1_train_labels.csv'
threshold = 50 #keep pixels at least this bright
min_area = 25
thresholds = [50]
intensity_threshold = 0.3
q_list = [0]
#min_areas = [12, 15, 20, 25, 30]


def temp_function():
    image_list = get_image_list(testing_data_path)
    results = []
    labeled_list = []
    for q in q_list:
        print("getting predictions")
        params = [threshold, intensity_threshold, q]
        labeled_list = run_basic_predict(image_list, params)
        print("removing tiny features")
        set_min_area(labeled_list, min_area)
        #print("scoring")
        #myscore = score_matrices(labeled_list, key_csv)
        #results.append((q,myscore[0]))
        #print(results[-1])
    for r in results:
        print(r)
    print("writing output")
    write_output_file(labeled_list, output_filename)



#cProfile.run('temp_function()')
temp_function()