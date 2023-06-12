from keras.models import Sequential
from keras.layers import Dense

# Create a Sequential model
model = Sequential()

# Add the first layer with 49 neurons and 'relu' activation function
model.add(Dense(49, input_dim=number_of_inputs, activation='relu'))

# Add the second layer with 49 neurons and 'relu' activation function
model.add(Dense(49, activation='relu'))

# Add the third layer with 49 neurons and 'relu' activation function
model.add(Dense(49, activation='relu'))

# Add the output layer with 'linear' activation function
model.add(Dense(1, activation='linear'))

# Compile the model with 'adam' optimizer, 'mean_squared_error' loss function, and 'accuracy' metric
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model for 250 epochs
model.fit(X_train, y_train, epochs=250, batch_size=32)
