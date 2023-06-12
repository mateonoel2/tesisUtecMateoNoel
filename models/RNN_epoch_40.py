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

        features = data[["day_of_week", "first_time", "total_distance", "first_stop", "target_stop"]]

        features = scaler.fit_transform(features)

        labels = data['label'].values

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False)

        X_test, X_final_test, y_test, y_final_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=False)

        # Reshape the input data to match the LSTM input shape (samples, timesteps, features)
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        X_final_test = np.reshape(X_final_test, (X_final_test.shape[0], 1, X_final_test.shape[1]))

        print(X_train)

        # Define the model
        model = load_model('../trained_models/RNN_90.h5')

        y_pred = model.predict(X_train)
        
        print(y_pred)

        # Convert X_test back to DataFrame
        X_test_df = pd.DataFrame(X_train.reshape(X_train.shape[0], X_train.shape[2]), columns=['day_of_week', 'first_time',  'total_distance', 'first_stop', 'target_stop'])

        # Convert y_pred to DataFrame
        y_pred_df = pd.DataFrame(y_pred, columns=['Predicted_Label'])

        # Concatenate the two dataframes along the columns
        result = pd.concat([X_test_df, y_pred_df], axis=1)

        print(result)

    except Exception as e:
        handle_error(e)