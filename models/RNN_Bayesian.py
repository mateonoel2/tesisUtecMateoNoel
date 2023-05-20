import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Sequential
from keras.layers import LSTM, Dense
from bayes_opt import BayesianOptimization


os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Define a function to handle errors during model training
def handle_error(e):
    print("Error occurred during model training: ", e)
    sys.exit(1)

def train_evaluate_model(epochs, batch_size, LSTM_units):
    try:
        LSTM_units = int(LSTM_units)
        epochs = int(epochs)
        batch_size = int(batch_size)
        
        data = pd.read_parquet("../unNormalized_dataset/dataset") 
        data = data.head(1000)
        
        scaler = StandardScaler()
        
        features = data[['day_of_week', 'total_speed', 'first_stop', 'target_stop', 'total_distance', 'first_time']]

        features = scaler.fit_transform(features)

        labels = data['label'].values

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, shuffle=False)

        # Reshape the input data to match the LSTM input shape (samples, timesteps, features)
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        # Define the model
        model = Sequential()
        model.add(LSTM(LSTM_units, input_shape=(1, features.shape[1])))
        model.add(Dense(1))

        # Compile the model
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Train the model
        model.fit(X_train, y_train, epochs=int(epochs), batch_size=int(batch_size), verbose=0)

        # Evaluate the model
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return -rmse

    except Exception as e:
        handle_error(e)

# Define the parameter ranges for Bayesian optimization
pbounds = {
    'LSTM_units': (32, 1024),
    'epochs': (10, 100),
    'batch_size': (16, 128),
}

# Define the Bayesian Optimizer object
optimizer = BayesianOptimization(
    f=train_evaluate_model,
    pbounds=pbounds,
    random_state=42,
    verbose=1
)

# Optimize the objective function
optimizer.maximize(init_points=40, n_iter=50)

# Print the best parameters and score
print(optimizer.max)