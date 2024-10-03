from definitions import *
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
import glob
import os

def load_images():
    images = glob.glob(f"{ROOT_DIR}\\datasets\\train\\*.jpg")
    labels = [os.path.basename(x).split('.')[0] for x in images]

    return images, labels

def split_training_data(images, labels):
    return sklearn.model_selection.train_test_split(
        images,
        labels,
        test_size=0.2,
        random_state=42)

def create_dataset(X_train, X_test, y_train, y_test):
    def package_data(X, y):
        unique = len(set(y))
        mapping = {label: i for i, label in enumerate(set(y))}

        y = [tf.one_hot(mapping[label], unique) for label in y]
        return X, y

    train = tf.data.Dataset.from_tensor_slices(package_data(X_train, y_train))
    test = tf.data.Dataset.from_tensor_slices(package_data(X_test, y_test))
    return train, test

def main():
    images, labels = load_images()
    X_trian, X_test, y_train, y_test = split_training_data(images, labels)
    train, test = create_dataset(X_trian, X_test, y_train, y_test)
    pass



if __name__ == '__main__':
    main()