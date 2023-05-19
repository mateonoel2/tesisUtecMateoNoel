import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from bayes_opt import BayesianOptimization
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define a function to handle errors during model training
def handle_error(e):
    print("Error occurred during model training: ", e)
    sys.exit(1)

def train_evaluate_model(epochs, batch_size):
    try:
        data = pd.read_parquet("../processed_datasets3/dataset")
        
        # Perform one-hot encoding for categorical features
        categorical_features = ['day_of_week', 'first_stop', 'target_stop']
        encoded_features = pd.get_dummies(data[categorical_features])
        numerical_features = data[['total_distance', 'first_time']]
        features = pd.concat([encoded_features, numerical_features], axis=1).values

        labels = data['label'].values

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, shuffle=False)

        # Reshape the input data to match the LSTM input shape (samples, timesteps, features)
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        # Define the model
        model = Sequential()
        model.add(LSTM(128, input_shape=(1, features.shape[1])))
        model.add(Dense(1))

        # Compile the model
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Train the model
        model.fit(X_train, y_train, epochs=int(epochs), batch_size=int(batch_size), verbose=4)

        # Evaluate the model
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return -rmse

    except Exception as e:
        handle_error(e)

# Define the parameter ranges for Bayesian optimization
pbounds = {
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
optimizer.maximize(init_points=5, n_iter=10)

# Print the best parameters and score
print(optimizer.max)