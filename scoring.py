from scoringUtil import score_output
import cProfile
import csv

training_data_path = 'data/stage1_train/'
output_csv = 'predictionBasic.csv'
key_csv = 'data/stage1_train_labels.csv'

myscore = score_output(output_csv, key_csv, training_data_path)
print(myscore[0])

with open('scoreperimage.csv', 'w') as csv_file:
    writer = csv.writer(csv_file, lineterminator='\n')
    writer.writerow(["ImageID", "Score"])
    for key, value in myscore[1].items():
        writer.writerow([str(key), str(value)])