from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import os
import json

# Directory path
data_dir = 'braintumor/Data/'

# Path to save the model
path_to_save_model = 'path_to_save_model'

# Create the directory if it does not exist
if not os.path.exists(path_to_save_model):
    os.makedirs(path_to_save_model)

# Data generator with validation split
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Flow images directly from the directory
train_generator = data_gen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',  # Change to categorical for multiple classes
    subset='training'  # Set as training data
)

# Flow images directly from the directory
valid_generator = data_gen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',  # Change to categorical for multiple classes
    subset='validation'  # Set as validation data
)

# Retrieve and print class indices
class_indices = train_generator.class_indices
print("Class Indices:", class_indices)

# Save class indices for later use
with open(os.path.join(path_to_save_model, 'class_indices.json'), 'w') as f:
    json.dump(class_indices, f)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')  # Number of classes
])

# Compile the model
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // valid_generator.batch_size
)

# Evaluate the model
loss, accuracy = model.evaluate(valid_generator)
print("Validation Accuracy:", accuracy)

# Save the model
model.save(os.path.join(path_to_save_model, 'model_name.keras'))
print(f"Model saved to {os.path.join(path_to_save_model, 'model_name.keras')}")