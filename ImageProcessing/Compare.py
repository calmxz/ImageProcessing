import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('path_to_save_model/model_name.keras')

# Load and preprocess the separate image
img_path = 'separate/single_prediction/cat_or_dog_2.jpg'
img = image.load_img(img_path, target_size=(150, 150))  # Adjust target_size if needed
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.

# Make predictions
predictions = model.predict(img_array)

# Classify the predictions
class_labels = ['cat', 'dog']  # Adjust based on your class labels
predicted_class = np.argmax(predictions)
predicted_label = class_labels[predicted_class]

# Get the probability associated with the predicted class
predicted_probability = predictions[0][predicted_class]

print("Predicted class:", predicted_label)
print("Accuracy: {:.2f}%".format(predicted_probability * 100))
