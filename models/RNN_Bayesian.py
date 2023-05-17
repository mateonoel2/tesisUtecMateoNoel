from pyspark.sql import SparkSession
from pyspark.sql.functions import abs, asc, col
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from bayes_opt import BayesianOptimization


# Define a function to handle errors during model training
def handle_error(e):
    print("Error occurred during model training: ", e)
    sys.exit(1)


def train_evaluate_model(epochs, batch_size):
    try:
        rmse_scores = []
        dataset_sizes = [int(56839426/89), int(56839426/45), int(56839426/22), int(56839426/11), int(56839426/5), int(56839426/2), int(56839426)]

        for dataset_size in dataset_sizes: 
            # Load data from Parquet
            data = spark.read.parquet("../processed_datasets3/dataset").coalesce(1)\
                .limit(dataset_size)\
                .orderBy(asc("month"), asc("day_of_month"), asc("exit_time")).cache()

            data = data.toPandas()

            # Perform one-hot encoding for categorical features
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
            model.fit(X_train, y_train, epochs=int(epochs), batch_size=int(batch_size), verbose=0)

            # Evaluate the model
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            rmse_scores.append(rmse)

        average_rmse = np.mean(rmse_scores)
        return -average_rmse 

    except Exception as e:
        handle_error(e)
    finally:
        spark.stop()


# Create a SparkSession
spark = SparkSession.builder \
    .appName("PySpark RNN LSTM") \
    .config("spark.executor.instances", "10") \
    .config("spark.executor.cores", "8") \
    .config("spark.driver.extraJavaOptions", "-XX:-UseGCOverheadLimit") \
    .config("spark.executor.extraJavaOptions", "-XX:-UseGCOverheadLimit") \
    .getOrCreate()
        
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
    verbose=2
)

# Optimize the objective function
optimizer.maximize(init_points=5, n_iter=10)

# Print the best parameters and score
print(optimizer.max)