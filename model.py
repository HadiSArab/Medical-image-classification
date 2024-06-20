import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd


data_dir = "F:\\tech\GitHub\Data\Alzeimer After Preprocessing"


categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]


image_data = {}


for category in categories:
    category_path = os.path.join(data_dir, category)
    for image_file in os.listdir(category_path):
        image = Image.open(os.path.join(category_path, image_file))
        image_data[os.path.join(category_path, image_file)] = category


image_paths = list(image_data.keys())
image_labels = np.array([image_data[path] for path in image_paths])


data = {'image_path': image_paths, 'label': image_labels}
df = pd.DataFrame(data)


print(df.head())


X_train1, X_test, y_train1, y_test = train_test_split(image_paths, image_labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=0.3, random_state=42)


# Initialize the model
model = Sequential()

# Add Convolutional and Pooling layers
model.add(Conv2D(16, kernel_size=(3,3), input_shape=(196, 196, 3), activation='relu'))
model.add(Conv2D(16, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())

# Add Dense layers
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(4))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))
