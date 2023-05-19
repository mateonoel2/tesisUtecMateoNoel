import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Define a function to handle errors during model training
def handle_error(e):
    print("Error occurred during model training: ", e)
    sys.exit(1)

if __name__ == '__main__':    
    try:
        data = pd.read_parquet("../processed_datasets3/dataset")
        #data = data.head(2000)

        for a in range(2):
            if a == 0:
                print("Short distances:")
                #Perform one-hot encoding for categorical features
                categorical_features = ['day_of_week', 'exit_stop', 'target_stop']
                encoded_features = pd.get_dummies(data[categorical_features])
                numerical_features = data[['distance', 'exit_time']]

            elif a == 1:
                print("Long distances:")
                #Perform one-hot encoding for categorical features
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
            model.fit(X_train, y_train, epochs=88, batch_size=23,  verbose=0)

            # Evaluate the model
            loss = model.evaluate(X_test, y_test)

            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Define the time thresholds in the normalized range
            ranges = [600/86400, 300/86400, 60/86400, 30/86400]
            
            y_pred = y_pred.flatten()

            for r in ranges:    
                diff = np.abs(y_pred - y_test)
                within_range = diff <= r
                print(np.sum(within_range))
                values_within_range = diff[within_range]
                percent_within_range = np.sum(within_range) / y_pred.shape[0] * 100
                print("{:.2f}% of data is within a range of {:.0f} seconds".format(percent_within_range, r * 86400))

            # Print metrics
            print("R-squared: {:.4f}".format(r2))
            print("RMSE: {:.4f}".format(rmse))
            
    except Exception as e:
        handle_error(e)