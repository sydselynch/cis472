import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import sklearn.neural_network
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


def train_nn(train_data):
    # split into train and validation sets
    labeled_images = pd.read_csv(train_data)
    images = labeled_images.iloc[0:10000,1:]
    labels = labeled_images.iloc[0:10000,:1]
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

    #standardize data
    s = StandardScaler()
    s.fit(train_images)
    train_images = s.transform(train_images)
    test_images = s.transform(test_images)

    nn = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(30,30,30))
    nn.fit(train_images, train_labels)
    predict = nn.predict(test_images)
    print(classification_report(test_labels,predict))

    return nn

def main(argv):
    if len(argv) != 2:
        print("usage: python digit_reader.py train.csv test.csv")
        sys.exit()
    training_data = argv[0]
    test_data = argv[1]

    # test NN
    test = pd.read_csv(test_data)
    trained_nn = train_nn(training_data)
    results = trained_nn.predict(test[0:])
    print("results: ")
    df = pd.DataFrame(results)
    df.index += 1
    df.index.name = 'ImageId'
    df.columns=['Label']
    df.to_csv('nn_results.csv', header=True)




if __name__ == "__main__":
    main(sys.argv[1:])
