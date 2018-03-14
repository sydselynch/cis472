import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm


def parse_data(file_name):
    with open(file_name, 'r') as f:
        r = csv.reader(f)
        d = list(r)
    variables = d[0]
    rows = d[1:]
    return (variables, rows)







def main(argv):
    variables, rows = parse_data(argv[0])

    # from kernel https://www.kaggle.com/archaeocharlie/a-beginner-s-approach-to-classification
    labeled_images = pd.read_csv(argv[0])
    images = labeled_images.iloc[0:5000,1:]
    labels = labeled_images.iloc[0:5000,:1]
    train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
    

    test_images[test_images>0]=1
    train_images[train_images>0]=1

    clf = svm.SVC()
    clf.fit(train_images, train_labels.values.ravel())
    print(clf.score(test_images,test_labels))

if __name__ == "__main__":
    main(sys.argv[1:])
