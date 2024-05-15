import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from data_collection.get_portfolio_data import \
    feature_data  # Ensure this import is correct based on your project structure


# Function to create LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Example shape: (number of time steps, number of features)
input_shape = (50, 3)

# Create model
model = create_lstm_model(input_shape)


def prepare_training_data(data, lookback=50):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 0])  # Assuming 'Close' is the first feature
    return np.array(X), np.array(y)


ticker = "NVDA"
data = feature_data[ticker][['Close', 'SMA_50', 'Volatility']].dropna().values

X, y = prepare_training_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))


def calculate_var(returns, confidence_level=0.95):
    var = np.percentile(returns, (1 - confidence_level) * 100)
    return var


def calculate_cvar(returns, var):
    cvar = returns[returns <= var].mean()
    return cvar


returns = np.diff(np.log(data[:, 0]))

var = calculate_var(returns)
cvar = calculate_cvar(returns, var)

print(f"VaR: {var}")
print(f"CVaR: {cvar}")
