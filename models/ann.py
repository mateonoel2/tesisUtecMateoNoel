# Import necessary libraries
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

# Load the data into a pandas dataframe
data = pd.read_csv('../processed_datasets/2014-08-01.csv')

# Convert date feature to datetime format
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

# Extract additional date features
data['day_of_week'] = data['date'].dt.dayofweek
data['day_of_month'] = data['date'].dt.day
data['month'] = data['date'].dt.month

data = data.drop('date', axis=1)

# Define a function to convert time strings to seconds since midnight
def time_to_seconds(time_str):
    dt = datetime.strptime(time_str, '%H:%M:%S')
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return (dt - midnight).total_seconds()

# Iterate over the 'exit_time' and 'arrive_time' columns and convert each value to seconds since midnight
for col in ['exit_time', 'arrive_time']:
    data[col] = data[col].apply(time_to_seconds)

# Concatenate the exit_stop and target_stop columns
stops = pd.concat([data['exit_stop'], data['target_stop']], ignore_index=True)

# Fit the encoder on the concatenated stops and transform the columns
le = LabelEncoder()
stops_encoded = le.fit_transform(stops)
data['exit_stop'] = stops_encoded[:len(data)]
data['target_stop'] = stops_encoded[len(data):]

# Scale numerical variables
scaler = MinMaxScaler()
data[['distance', 'exit_time']] = scaler.fit_transform(data[['distance', 'exit_time']])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('arrive_time', axis=1), data['arrive_time'], test_size=0.2)

# Define the architecture of the model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
early_stop = EarlyStopping(monitor='val_loss', patience=10)
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop])

# Evaluate the model
mse = model.evaluate(X_test, y_test)
rmse = np.sqrt(mse)
print('Root Mean Squared Error:', rmse)