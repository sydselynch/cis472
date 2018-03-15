import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten
from keras.optimizers import Adam, RMSprop

def parse_data(file_name):
    with open(file_name, 'r') as f:
        r = csv.reader(f)
        d = list(r)
    variables = d[0]
    rows = d[1:]
    return (variables, rows)


def support_vector_machine_train(training_data):
    '''
    svm
    from kernel https://www.kaggle.com/archaeocharlie/a-beginner-s-approach-to-classification
    '''
    labeled_images = pd.read_csv(training_data)
    images = labeled_images.iloc[0:10000,1:]
    labels = labeled_images.iloc[0:10000,:1]
    train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

    # convert all pixels to black and white
    test_images[test_images>0]=1
    train_images[train_images>0]=1

    # can tune these parameters
    # see http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    clf = svm.SVC(C=7, kernel='rbf', gamma=0.009)
    clf.fit(train_images, train_labels.values.ravel())
    print("Accuracy: ", clf.score(test_images,test_labels))

    return clf

def main(argv):
    if len(argv) != 2:
        print("usage: python digit_reader.py train.csv test.csv")
        sys.exit()
    training_data = argv[0]
    test_data = argv[1]



    # test svm
    clf = support_vector_machine_train(training_data)
    test = pd.read_csv(test_data)
    test[test>0]=1
    results = clf.predict(test[0:])
    print(results)
    df = pd.DataFrame(results)
    df.index += 1
    df.index.name = 'ImageId'
    df.columns=['Label']
    df.to_csv('results.csv', header=True)



if __name__ == "__main__":
    main(sys.argv[1:])
