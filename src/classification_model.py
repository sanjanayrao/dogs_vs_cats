# Importing all necessary libraries
from keras import layers, models, optimizers, utils


training_dir = "./datasets/train"
testing_dir = "./datasets/test"
if __name__ == "__main__":

    # sample model, TODO need to tweak
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(learning_rate=1e-4),
                  metrics=['acc'])



    # load datasets
    training_dataset = utils.image_dataset_from_directory(
        directory="./datasets/train",
        labels="inferred",
        color_mode="rgb",
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=False,
        data_format=None,
        verbose=True,
    )

    validation_dataset = utils.image_dataset_from_directory(
        directory="./datasets/test",
        labels=None,
        color_mode="rgb",
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=False,
        data_format=None,
        verbose=True,
    )

    # train model
    history = model.fit(


    )

    # fit_generator(
    #     train_generator,
    #     steps_per_epoch=320,  # 50 batches in the generator, so it takes 320 batches to get to 16000 images
    #     epochs=30,
    #     validation_data=validation_generator,
    #     validation_steps=90)  # 90 x 50 == 4500