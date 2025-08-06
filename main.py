import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load historical S&P 500 data (you'll need to provide your own CSV file)
data = pd.read_csv('sp500.csv', parse_dates=['Date'], index_col='Date')
data = data[['Close']].dropna()

# Parameters
window = 20
threshold = 0.02
initial_capital = 10000
position_size = 0.1  # percent of capital

# Rolling indicators
data['SMA'] = data['Close'].rolling(window=window).mean()
data['Signal'] = 0
data['Signal'][window:] = np.where(
    (data['Close'][window:] > (1 + threshold) * data['SMA'][window:]), 1,
    np.where((data['Close'][window:] < (1 - threshold) * data['SMA'][window:]), -1, 0)
)
data['Position'] = data['Signal'].shift(1)

# Backtesting logic
capital = initial_capital
positions = []
capital_history = []
shares = 0

for date, row in data.iterrows():
    price = row['Close']
    signal = row['Position']
    
    if signal == 1 and capital > 0:
        allocation = capital * position_size
        shares = allocation // price
        capital -= shares * price
    elif signal == -1 and shares > 0:
        capital += shares * price
        shares = 0

    total_value = capital + shares * price
    capital_history.append(total_value)
    positions.append(shares)

data['Capital'] = capital_history

data['Returns'] = data['Capital'].pct_change()
data['Drawdown'] = (data['Capital'] / data['Capital'].cummax()) - 1

# Final return
final_return = (data['Capital'].iloc[-1] - initial_capital) / initial_capital * 100
print(f"Final Return: {final_return:.2f}%")

# Plotting
plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1)
plt.plot(data['Capital'], label='Portfolio Value')
plt.title('Trading Strategy Performance')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(data['Drawdown'], label='Drawdown', color='red')
plt.title('Drawdown')
plt.legend()

plt.tight_layout()
plt.show()

# Sensitivity analysis (varying thresholds)
returns = []
thresholds = np.linspace(0.01, 0.05, 10)

for thresh in thresholds:
    data['Signal'] = 0
    data['Signal'][window:] = np.where(
        (data['Close'][window:] > (1 + thresh) * data['SMA'][window:]), 1,
        np.where((data['Close'][window:] < (1 - thresh) * data['SMA'][window:]), -1, 0)
    )
    data['Position'] = data['Signal'].shift(1)

    capital = initial_capital
    shares = 0
    capital_history = []

    for date, row in data.iterrows():
        price = row['Close']
        signal = row['Position']

        if signal == 1 and capital > 0:
            allocation = capital * position_size
            shares = allocation // price
            capital -= shares * price
        elif signal == -1 and shares > 0:
            capital += shares * price
            shares = 0

        total_value = capital + shares * price
        capital_history.append(total_value)

    ret = (capital_history[-1] - initial_capital) / initial_capital
    returns.append(ret)

# Plot sensitivity analysis
plt.figure(figsize=(8, 4))
plt.plot(thresholds, returns, marker='o')
plt.title('Parameter Sensitivity: Threshold vs Return')
plt.xlabel('Threshold')
plt.ylabel('Return')
plt.grid(True)
plt.show()
