import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def train_and_evaluate_model(merged_data):
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(merged_data)

    # Create sequences for LSTM
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i+seq_length]
            y = data[i+seq_length, 0]  # Assuming the target variable is the first column
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    seq_length = 60  # Number of time steps to look back
    X, y = create_sequences(scaled_data, seq_length)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')

    # Predict and inverse transform the predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # Compare predictions with actual values
    actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plotting the results

    plt.figure(figsize=(14, 5))
    plt.plot(actual_values, color='blue', label='Actual Values')
    plt.plot(predictions, color='red', label='Predicted Values')
    plt.title('Prediction vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
