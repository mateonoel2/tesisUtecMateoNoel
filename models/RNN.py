from pyspark.sql import SparkSession
from pyspark.sql.functions import abs, asc, col
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
        # Create a SparkSession
        spark = SparkSession.builder \
        .appName("PySpark RNN LSTM")\
        .config("spark.executor.instances", "10")\
        .config("spark.executor.cores", "8")\
        .config("spark.driver.extraJavaOptions", "-XX:-UseGCOverheadLimit") \
        .config("spark.executor.extraJavaOptions", "-XX:-UseGCOverheadLimit") \
        .getOrCreate()
        
        #total_count_data = 10000
        print("10000 rows in 7am to 9am")

        # Load data from Parquet
        data = spark.read.parquet("../processed_datasets3/dataset").coalesce(1)\
                .filter((col("label") >= 0.291) & (col("label") <= 0.375))\
                .orderBy(asc("month"), asc("day_of_month"), asc("exit_time")).cache()
                #.filter((col("label") >= 0.291) & (col("label") <= 0.375))\
                #.limit(total_count_data)\
                #.filter((col("month") == 8) & (col("day_of_month") <= 3))\
        
        data = data.toPandas()

        #Perform one-hot encoding for categorical features
        categorical_features = ['day_of_week', 'exit_stop', 'target_stop']
        encoded_features = pd.get_dummies(data[categorical_features])
        numerical_features = data[['distance', 'exit_time']]
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
        model.fit(X_train, y_train, epochs=10, batch_size=32)

        # Evaluate the model
        loss = model.evaluate(X_test, y_test)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Define the time thresholds in the normalized range
        thresholds = [30/86400, 60/86400, 300/86400, 600/86400]

        # Convert labels and predictions to binarized values
        y_true_bins = np.digitize(y_test, thresholds)
        y_pred_bins = np.digitize(y_pred.flatten(), thresholds)

        # Define the time thresholds in the normalized range
        time_ranges = ['30s', '1m', '5m', '10m']

        accuracies = []
        for i in range(1, len(thresholds) + 1):
            true_mask = y_true_bins == i
            pred_mask = y_pred_bins == i

            print("Time Range {}: Number of true samples: {}".format(time_ranges[i-1], np.sum(true_mask)))
            print("Time Range {}: Number of predicted samples: {}".format(time_ranges[i-1], np.sum(pred_mask)))

            # Check if any samples fall within the time range
            if np.any(true_mask):
                accuracy = accuracy_score(y_true_bins[true_mask], y_pred_bins[true_mask])
                accuracies.append(accuracy)
            else:
                accuracies.append(0.0)

        # Print metrics
        print("R-squared: {:.4f}".format(r2))
        print("RMSE: {:.4f}".format(rmse))
        for i, accuracy in enumerate(accuracies):
            print("Accuracy in Time Range {}: {:.2f}%".format(time_ranges[i], accuracy * 100))

    except Exception as e:
        handle_error(e)
    finally:
        spark.stop()