from scoringUtil import score
import cProfile

training_data_path = 'data/stage1_train/'
output_csv = 'predictionTrainLower.csv'
key_csv = 'data/stage1_train_labels.csv'

test1 = cProfile.run('score(output_csv, key_csv, training_data_path)')
print('hi')
#myscore = score(output_csv, key_csv, training_data_path)
#print(myscore[0])