from keras import utils, Sequential
import tensorflow as tf
import matplotlib.pyplot as plt

# Path to your images
image_path = './datasets/test/'  # Adjust for the image format you're using
test_dataset = utils.image_dataset_from_directory(
        directory=image_path,
        color_mode="rgb",
        labels=None,
        shuffle=True,
        image_size=(64, 64),
        interpolation="bicubic",  # for sharper images
        crop_to_aspect_ratio=True,
        verbose=True,
    )

for img_batch in test_dataset:
    print(f'Batch shape: {img_batch.shape}')
    break  # Print only the first batch

my_model = tf.keras.models.load_model('test_model.keras')
predictions = my_model.predict(test_dataset)
labels = ('cat', 'dog')
threshold = 0.5
for i, pred in enumerate(predictions):
    predicted_class = "dog" if pred[0] >= threshold else "cat"
    print(f'Image {i + 1}: Predicted class: {predicted_class}, Probability: {pred[0]}')
plt.hist(predictions, bins=50)
plt.title('Distribution of Prediction Probabilities')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.show()