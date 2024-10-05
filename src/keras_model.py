from definitions import *
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.client import device_lib
import keras
from keras import losses
from keras import ops
from keras import optimizers
from keras import metrics
import keras_cv
import glob
import os
import math

BATCH_SIZE = 32

def load_images():
    images = glob.glob(f"{ROOT_DIR}\\datasets\\train\\*.jpg")
    labels = [os.path.basename(x).split('.')[0] for x in images]

    return images, labels

def split_training_data(images, labels, test_size=0.2):
    return sklearn.model_selection.train_test_split(
        images,
        labels,
        test_size=test_size,
        random_state=42)

def create_dataset(X_train, X_test, y_train, y_test):
    def package_data(X, y):
        unique = len(set(y))
        mapping = {label: i for i, label in enumerate(set(y))}

        X = [keras.utils.img_to_array(keras.utils.load_img(image, target_size=(128,128)), dtype='uint8') for image in X]
        y = [tf.one_hot(mapping[label], unique) for label in y]
        return X, y

    def map_data(image, label):
        return {"images": image, "labels": label}

    train = tf.data.Dataset.from_tensor_slices(package_data(X_train, y_train)).map(map_data)
    test = tf.data.Dataset.from_tensor_slices(package_data(X_test, y_test)).map(map_data)
    return train, test

def main():
    print(device_lib.list_local_devices())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    with tf.device('/gpu:0'):
        images, labels = load_images()
        X_train, X_test, y_train, y_test = split_training_data(images, labels, test_size=0.95)
        X_train, X_test, y_train, y_test = split_training_data(X_train, y_train)
        train, test = create_dataset(X_train, X_test, y_train, y_test)

        train = train.shuffle(BATCH_SIZE * 16)

        train = train.ragged_batch(BATCH_SIZE)
        test = test.ragged_batch(BATCH_SIZE)

        batch = next(iter(train.take(1)))
        image_batch = batch["images"]
        label_batch = batch["labels"]

        def unpackage_dict(inputs):
            return inputs["images"], inputs["labels"]

        train = train.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)
        test = test.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)

        backbone = keras_cv.models.EfficientNetV2B0Backbone()
        model = keras.Sequential(
            [
                backbone,
                keras.layers.GlobalMaxPooling2D(),
                keras.layers.Dropout(rate=0.5),
                keras.layers.Dense(2, activation="softmax"),
            ]
        )

        def lr_warmup_cosine_decay(
                global_step,
                warmup_steps,
                hold=0,
                total_steps=0,
                start_lr=0.0,
                target_lr=1e-2,
        ):
            # Cosine decay
            learning_rate = (
                    0.5
                    * target_lr
                    * (
                            1
                            + ops.cos(
                        math.pi
                        * ops.convert_to_tensor(
                            global_step - warmup_steps - hold, dtype="float32"
                        )
                        / ops.convert_to_tensor(
                            total_steps - warmup_steps - hold, dtype="float32"
                        )
                    )
                    )
            )

            warmup_lr = target_lr * (global_step / warmup_steps)

            if hold > 0:
                learning_rate = ops.where(
                    global_step > warmup_steps + hold, learning_rate, target_lr
                )

            learning_rate = ops.where(global_step < warmup_steps, warmup_lr, learning_rate)
            return learning_rate

        class WarmUpCosineDecay(optimizers.schedules.LearningRateSchedule):
            def __init__(self, warmup_steps, total_steps, hold, start_lr=0.0, target_lr=1e-2):
                super().__init__()
                self.start_lr = start_lr
                self.target_lr = target_lr
                self.warmup_steps = warmup_steps
                self.total_steps = total_steps
                self.hold = hold

            def __call__(self, step):
                lr = lr_warmup_cosine_decay(
                    global_step=step,
                    total_steps=self.total_steps,
                    warmup_steps=self.warmup_steps,
                    start_lr=self.start_lr,
                    target_lr=self.target_lr,
                    hold=self.hold,
                )

                return ops.where(step > self.total_steps, 0.0, lr)

        total_images = 250
        total_steps = (total_images // BATCH_SIZE) * 1
        warmup_steps = int(0.1 * total_steps)
        hold_steps = int(0.45 * total_steps)
        schedule = WarmUpCosineDecay(
            start_lr=0.05,
            target_lr=1e-2,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            hold=hold_steps,
        )
        optimizer = optimizers.SGD(
            weight_decay=5e-4,
            learning_rate=schedule,
            momentum=0.9,
        )

        model.compile(
            loss=losses.CategoricalCrossentropy(label_smoothing=0.1),
            optimizer='adam',
            metrics=[
                metrics.CategoricalAccuracy(),
                metrics.TopKCategoricalAccuracy(k=5),
            ],
        )

        model.fit(
            train,
            epochs=32,
            validation_data=test,
        )





if __name__ == '__main__':
    main()