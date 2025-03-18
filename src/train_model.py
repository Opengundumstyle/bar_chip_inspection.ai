import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load dataset
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = datagen.flow_from_directory("../dataset/", target_size=(128, 128), batch_size=32, class_mode='binary', subset='training')
val_data = datagen.flow_from_directory("../dataset/", target_size=(128, 128), batch_size=32, class_mode='binary', subset='validation')

# Train model
model.fit(train_data, validation_data=val_data, epochs=10)

# Save model
model.save("../models/defect_detector.h5")
print("Model saved!")
