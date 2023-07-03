import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import load_model

# Define a function to handle errors during model training
def handle_error(e):
    print("Error occurred during model training: ", e)
    sys.exit(1)

if __name__ == '__main__':    
    try:
        data = pd.read_parquet("../unNormalized_dataset/dataset") 

        scaler = pickle.load(open('../trained_models/rnn_scaler.pkl', 'rb'))
        
        features = data[["day_of_week", "first_time", "total_distance", "first_stop", "target_stop"]]

        print(features)

        features = scaler.transform(features)

        labels = data['label'].values
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False)

        # Reshape the input data to match the LSTM input shape (samples, timesteps, features)
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        model = load_model('../trained_models/RNN_90.h5')

        total_epochs = 100000

        min_loss = np.inf
     
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Define the time thresholds in the range
        ranges = [600, 300, 60, 30]
        
        y_pred = y_pred.flatten()

        for r in ranges:    
            diff = np.abs(y_pred - y_test)
            within_range = diff <= r
            values_within_range = diff[within_range]
            percent_within_range = np.sum(within_range) / y_pred.shape[0] * 100
            print("{:.2f}% of data is within a range of {:.0f} seconds".format(percent_within_range, r))

        # Print metrics
        print("R-squared: {:.4f}".format(r2))
        print("RMSE: {:.4f}".format(rmse))

    except Exception as e:
        handle_error(e)