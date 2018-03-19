import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from timeit import default_timer as timer

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

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
    train = pd.read_csv(train_data)
    Y_train = train["label"]
    X_train = train.drop(labels = ["label"], axis=1)
    X_train = X_train.values.reshape(-1,28,28,1)
    Y_train = to_categorical(Y_train, num_classes = 10)

    train_images, test_images, train_labels, test_labels = train_test_split(X_train, Y_train, train_size=0.9, random_state=0)

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



    nn = Sequential()

    nn.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                     activation ='relu', input_shape = (28,28,1)))
    nn.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                     activation ='relu'))
    nn.add(MaxPool2D(pool_size=(2,2)))
    nn.add(Dropout(0.25))


    nn.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                     activation ='relu'))
    nn.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                     activation ='relu'))
    nn.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    nn.add(Dropout(0.25))


    nn.add(Flatten())
    nn.add(Dense(256, activation = "relu"))
    nn.add(Dropout(0.5))
    nn.add(Dense(10, activation = "softmax"))

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    nn.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

    epochs = 30 # Turn epochs to 30 to get 0.9967 accuracy
    batch_size = 86

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


    datagen.fit(train_images)
    history = nn.fit_generator(datagen.flow(train_images, train_labels, batch_size=batch_size),
                              epochs = epochs, validation_data = (test_images,test_labels),
                              verbose = 2, steps_per_epoch=train_images.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])



    return nn

def main(argv):
    if len(argv) != 2:
        print("usage: python digit_reader.py train.csv test.csv")
        sys.exit()
    training_data = argv[0]
    test_data = argv[1]

    # test NN
    test = pd.read_csv(test_data)
    test = test / 255.0
    test = test.values.reshape(-1,28,28,1)
    start = timer()
    trained_nn = train_nn(training_data)
    duration = timer() - start
    print("Training duration: ", duration)

    results = trained_nn.predict(test)
    df = pd.DataFrame(results)
    df.index += 1
    df.index.name = 'ImageId'
    df.columns=['Label']
    df.to_csv('nn_keras_results.csv', header=True)




if __name__ == "__main__":
    main(sys.argv[1:])
