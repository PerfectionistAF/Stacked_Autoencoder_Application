import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# Assuming joined_input is your 3D input array with shape (num_samples, height, width, channels)
# For example, joined_input.shape = (num_samples, 256, 256, 3)

# Flatten the image array
flattened_input = joined_input.reshape(joined_input.shape[0], -1)
print(flattened_input[0:5])

# Define the autoencoder architecture
input_dim = flattened_input.shape[1]  # Number of features after flattening
hidden_size = 100  # Size of the hidden layer

# Input layer for the dense part
input_layer = Input(shape=(input_dim,))

# Encoder (Dense layer)
encoder = Dense(hidden_size, activation='relu', 
                activity_regularizer=tf.keras.regularizers.l1(4e-4),
                kernel_regularizer=l2(0.004))(input_layer)

# Reshape the dense layer output back to 4D shape
reshaped_encoder = Reshape((16, 16, hidden_size // (16 * 16)))(encoder)  # Adjust the shape as needed

# Convolutional layers
x = Conv2D(32, (3, 3), activation='relu', padding='same')(reshaped_encoder)
x = Lambda(lambda x: tf.nn.local_response_normalization(x))(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Lambda(lambda x: tf.nn.local_response_normalization(x))(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = Lambda(lambda x: tf.nn.local_response_normalization(x))(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

decoded = Conv2D(joined_input.shape[3], (3, 3), activation='sigmoid', padding='same')(x)

# Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoded)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
early_stopping = EarlyStopping(monitor='loss', patience=10)
autoencoder.fit(flattened_input, joined_input,  # Autoencoder targets are the original images
                epochs=400,
                batch_size=256,
                shuffle=True,
                callbacks=[early_stopping])

# View the model architecture
autoencoder.summary()

# Extract features from the encoder part of the autoencoder
encoder_model = Model(inputs=input_layer, outputs=encoder)
features = encoder_model.predict(flattened_input)

# Reshape the features to (616, 128, 384)
reshaped_features = features.reshape(616, 128, 384)

# Plot the features
plt.figure(figsize=(10, 10))
for i in range(min(reshaped_features.shape[0], 100)):  # Plot up to 100 feature maps
    plt.subplot(10, 10, i + 1)
    plt.imshow(reshaped_features[i, :, :], cmap='viridis')  # Plot the first channel
    plt.axis('off')
plt.suptitle('Features from the Encoder')
plt.show()