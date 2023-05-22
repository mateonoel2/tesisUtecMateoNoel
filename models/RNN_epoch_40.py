import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model

# Define a function to handle errors during model training
def handle_error(e):
    print("Error occurred during model training: ", e)
    sys.exit(1)

if __name__ == '__main__':    
    try:
        data = pd.read_parquet("../unNormalized_dataset/dataset") 

        scaler = StandardScaler()
        
        print("Long distances:")

        features = data[['day_of_week', 'first_stop', 'target_stop', 'total_distance', 'first_time']]

        features = scaler.fit_transform(features)

        labels = data['label'].values

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False)

        X_test, X_final_test, y_test, y_final_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=False)

        # Reshape the input data to match the LSTM input shape (samples, timesteps, features)
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        X_final_test = np.reshape(X_final_test, (X_final_test.shape[0], 1, X_final_test.shape[1]))

        # Define the model
        model = load_model('../trained_models/RNN_40.h5')

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

        y_pred = model.predict(X_final_test)

        r2 = r2_score(y_final_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_final_test, y_pred))

        # Define the time thresholds in the range
        ranges = [600, 300, 60, 30]
        
        y_pred = y_pred.flatten()

        for r in ranges:    
            diff = np.abs(y_pred - y_final_test)
            within_range = diff <= r
            values_within_range = diff[within_range]
            percent_within_range = np.sum(within_range) / y_pred.shape[0] * 100
            print("{:.2f}% of data is within a range of {:.0f} seconds".format(percent_within_range, r))

        # Print metrics
        print("R-squared: {:.4f}".format(r2))
        print("RMSE: {:.4f}".format(rmse))

        model.save(f'../trained_models/RNN_35.h5')

    except Exception as e:
        handle_error(e)