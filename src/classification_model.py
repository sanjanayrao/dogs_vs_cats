from collections import Counter

from keras import utils, Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D

training_dir = "./datasets/train"
if __name__ == "__main__":

    # load datasets
    training_dataset = utils.image_dataset_from_directory(
        directory=training_dir,
        labels="inferred",
        color_mode="rgb",
        shuffle=True,
        image_size=(256, 256),
        interpolation="bicubic",  # for sharper images
        crop_to_aspect_ratio=True,
        verbose=True,
    )
    val_split = 0.2
    train_size = int((1 - val_split) * len(training_dataset))

    train_dataset = training_dataset.skip(train_size)
    val_dataset = training_dataset.take(train_size)
    # all_labels = []
    #
    # for images, labels in training_dataset:
    #     all_labels.extend(labels.numpy())
    #
    # label_counts = Counter(all_labels)
    # print(training_dataset.class_names)
    #
    # print(label_counts)

    model = Sequential([
        Conv2D(12, (3, 3), activation='relu', input_shape=(256, 256, 3)),  # Increased filters
        MaxPooling2D(),
        BatchNormalization(),

        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(),
        BatchNormalization(),

        Conv2D(64, (3, 3), activation='relu'),  # Added more filters
        MaxPooling2D(),
        BatchNormalization(),

        GlobalAveragePooling2D(),

        Dense(128, activation='relu'),  # Increased size of dense layer
        Dropout(0.2),  # Increased dropout for regularization
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(
        train_dataset,
        epochs=10,
        validation_data=val_dataset,
        verbose=1,
        batch_size=32
    )


    model.save('test_model.keras')
