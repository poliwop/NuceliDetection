from scoringUtil import score

training_data_path = 'data/stage1_train/'
output_csv = 'predictionTrain.csv'

myscore = score(output_csv, training_data_path)
print(myscore[0])