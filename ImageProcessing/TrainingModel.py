import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

# Step 1: Prepare your dataset
train_dir = 'datasets/train'
test_dir = 'datasets/test'
batch_size = 32
image_size = (150, 150)

# Step 2: Load the dataset
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical')  # Use 'categorical' for multi-class classification

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical')

# Step 3: Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(2, activation='softmax')  # 2 output neurons for apple and banana
])

# Step 4: Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Use categorical crossentropy for multi-class classification
              metrics=['accuracy'])

# Step 5: Train the model
history = model.fit(
      train_generator,
      steps_per_epoch=train_generator.samples // batch_size,
      epochs=1,
      validation_data=test_generator,
      validation_steps=test_generator.samples // batch_size)

# Step 6: Evaluate the model
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print('\nTest accuracy:', test_acc)

# Step 7: Save the model
directory = 'path_to_save_model'  # this is optional model

# Ensure that the directory and its parent directories exist
os.makedirs(directory, exist_ok=True)

# Save the model in the native Keras format
model.save(os.path.join(directory, 'model_name.keras'))

print("Model saved successfully.")