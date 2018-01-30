from scoringUtil import score_output
import cProfile

training_data_path = 'data/stage1_train/'
output_csv = 'predictionBasic.csv'
key_csv = 'data/stage1_train_labels.csv'

myscore = score_output(output_csv, key_csv, training_data_path)
print(myscore[0])