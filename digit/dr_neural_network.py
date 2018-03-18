import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer


import sklearn.neural_network
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np

from keras.preprocessing.image import ImageDataGenerator


from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)


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
    images = labeled_images.iloc[0:,1:]
    labels = labeled_images.iloc[0:,:1]
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.9, random_state=0)

    #standardize data
    # Multiple ways to normalize data

    # s = StandardScaler()
    # s.fit(train_images)
    # train_images = s.transform(train_images)
    # test_images = s.transform(test_images)


    # test_images[test_images>0]=1  convert to black and white from grayscale
    # train_images[train_images>0]=1

    test_images = test_images / 255.0
    train_images = train_images / 255.0

    # randomize images
    


    neural_network = sklearn.neural_network.MLPClassifier(activation='relu', solver='adam', hidden_layer_sizes=(500,))
    neural_network.fit(train_images, train_labels)
    predict = neural_network.predict(test_images)
    print(neural_network.score(test_images, test_labels))
    print(classification_report(test_labels,predict,digits=10))

    return neural_network

def main(argv):
    if len(argv) != 2:
        print("usage: python digit_reader.py train.csv test.csv")
        sys.exit()
    training_data = argv[0]
    test_data = argv[1]

    # test NN
    test = pd.read_csv(test_data)
    start = timer()
    trained_nn = train_nn(training_data)
    duration = timer() - start
    print("Training duration: ", duration)

    results = trained_nn.predict(test[0:])
    df = pd.DataFrame(results)
    df.index += 1
    df.index.name = 'ImageId'
    df.columns=['Label']
    df.to_csv('nn_results.csv', header=True)




if __name__ == "__main__":
    main(sys.argv[1:])
