import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import time
import winsound


def parse_data(file_name):
    with open(file_name, 'r') as f:
        r = csv.reader(f)
        d = list(r)
    variables = d[0]
    rows = d[1:]
    return (variables, rows)


def knn_train(training_data):
    '''
    knn-classifier
    from kernel https://www.kaggle.com/benjamind123/knn-classifier
    '''
    labeled_images = pd.read_csv(training_data)
    images = labeled_images.iloc[0:10000,1:]
    labels = labeled_images.iloc[0:10000,:1]
    train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

    # convert all pixels to black and white
    test_images[test_images>0]=1
    train_images[train_images>0]=1
    
    # creating odd list of K for KNN
    myList = list(range(1,10))

    # subsetting just the odd ones
    neighbors = filter(lambda x: x % 2 != 0, myList)

    loo = LeaveOneOut()
    n = loo.get_n_splits(train_images)
    print(n)
    maxacc = 0
    maxk = 1
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, train_images, train_labels.values.ravel(), cv=n, scoring='accuracy')
        accuracy = scores.mean()
        print("CV score K=",k, ": ", accuracy)
        if accuracy > maxacc:
            maxacc = accuracy
            maxk = k
        
        

    start = time.time()
    print("Fitting k = ", maxk)
    knn = KNeighborsClassifier(n_neighbors=maxk)
    knn.fit(train_images, train_labels.values.ravel())
    end = time.time()
    print("Training time: ", end - start)

    return knn

def main(argv):
    if len(argv) != 2:
        print("usage: python dr_knn.py train.csv test.csv")
        sys.exit()
    training_data = argv[0]
    test_data = argv[1]

    knn = knn_train(training_data)
#    test = pd.read_csv(test_data)
#    test[test>0]=1
#    start = time.time()
#    results = knn.predict(test)
#    end = time.time()
#    print("Predicting time: ", end - start)
#    print(results)
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    for i in range(3):
        winsound.Beep(frequency, duration)
#    df = pd.DataFrame(results)
#    df.index += 1
#    df.index.name = 'ImageId'
#    df.columns=['Label']
#    df.to_csv('knn9_results.csv', header=True)



if __name__ == "__main__":
    main(sys.argv[1:])
