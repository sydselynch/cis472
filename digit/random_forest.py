import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np


def parse_data(file_name):
	with open(file_name, 'r') as f:
		r = csv.reader(f)
		d = list(r)
	variables = d[0]
	rows = d[1:]
	return (variables, rows)




def main(argv):
	if len(argv) != 2:
		print("usage: python digit_reader.py train.csv test.csv")
		sys.exit()
	training_data = argv[0]
	test_data = argv[1]
	# data = pd.read_csv(training_data)
	test = pd.read_csv(test_data)
	#
	# X = data.drop("label", axis=1)
	# Y = data.label

	labeled_images = pd.read_csv(training_data)
	images = labeled_images.iloc[0:10000,1:]
	labels = labeled_images.iloc[0:10000,:1]
	train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

	#standardize data
	s = StandardScaler()
	s.fit(train_images)
	train_images = s.transform(train_images)
	test_images = s.transform(test_images)

	model = RandomForestClassifier(n_estimators = 1000, n_jobs=-1)
	model.fit(train_images, train_labels)
	predict = model.predict(test_images)
	print(classification_report(test_labels,predict))


	#print(model.score(X, Y))
	results = model.predict(test)
	sub = pd.DataFrame({'ImageId':test.index +1,'Label':results})
	sub[['ImageId','Label']].to_csv('submission.csv', index=False)




if __name__ == "__main__":
	main(sys.argv[1:])
