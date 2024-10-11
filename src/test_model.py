import glob
import tensorflow as tf
import matplotlib.pyplot as plt

# Path to your images
image_path = 'datasets/test/*.jpg'  # Adjust for the image format you're using

# Get list of image file paths
image_files = glob.glob(image_path)

# Create a dataset from the image files
test_dataset = tf.data.Dataset.from_tensor_slices(image_files).take(100)
def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # Use appropriate decode function
    img = tf.image.resize(img, [256, 256])  # Resize to match model input
    img /= 255.0  # Normalize
    return img

# Map the loading function to the dataset
test_dataset = test_dataset.map(load_and_preprocess_image).batch(32)

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
# plt.hist(predictions, bins=50)
# plt.title('Distribution of Prediction Probabilities')
# plt.xlabel('Probability')
# plt.ylabel('Frequency')
# plt.show()