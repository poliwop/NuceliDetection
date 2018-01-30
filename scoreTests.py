from scoringUtil import score_output
import os



out_path = 'testOut/'
file_list = os.listdir(out_path)
training_data_path = 'data/stage1_train/'
key_csv = 'data/stage1_train_labels.csv'
scores = [None]*len(file_list)
for i,f in enumerate(file_list):
    print(f)
    scores[i] = score_output(out_path + f, key_csv, training_data_path)
    print(scores[i][0])

print(scores[0][0])