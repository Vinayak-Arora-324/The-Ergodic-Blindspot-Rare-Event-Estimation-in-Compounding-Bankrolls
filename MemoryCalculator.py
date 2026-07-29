import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression


df = pd.read_csv('chart.csv')

# Display basic information about the data
# print("First few rows of the data:")
# print(df.head())

# print("\nData types:")
# print(df.dtypes)

#print("\nSummary statistics:")
#print(df.describe())

# Extract specific columns
dates = df['DateTime']
prices = df['Cotton Prices - 45 Year Historical Chart']


returns = np.log(prices).diff().dropna()
## Rescaled Range (R/S) Analysis
def hurst_rs(series, max_lag=50):
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]

H_rs = hurst_rs(returns.values)
print("Rescaled range analysis")
print(f"Hurst (R/S): {H_rs:.4f}")


## Detrended Fluctuation Analysis (DFA)
def hurst_dfa(series, max_lag=50):
    n = len(series)
    lags = range(10, max_lag)
    F = []
    for lag in lags:
        # Split into chunks
        chunks = [series[i:i + lag] for i in range(0, n, lag)]
        # Detrend each chunk (remove linear trend)
        rms = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            x = np.arange(len(chunk))
            y = chunk
            coeffs = np.polyfit(x, y, 1)
            trend = np.polyval(coeffs, x)
            detrended = y - trend
            rms.append(np.sqrt(np.mean(detrended**2)))
        F.append(np.mean(rms))
    # Fit log-log plot
    poly = np.polyfit(np.log(lags), np.log(F), 1)
    return poly[0]

H_dfa = hurst_dfa(returns.values)
print("Detrended Fluctuation Analysis (DFA)")
print(f"Hurst (DFA): {H_dfa:.4f}")

def higuchi_fd(series, k_max=10):
    n = len(series)
    L = []
    for k in range(1, k_max + 1):
        Lk = 0
        for m in range(k):
            # Take every k-th element starting at m
            x = series[m::k]
            # Compute the length of the curve
            Lkm = np.sum(np.abs(np.diff(x))) * (n - 1) / (len(x) * k)
            Lk += Lkm
        L.append(np.log(Lk / k))
    # Fit a line to the log-log plot
    x = np.log(np.arange(1, k_max + 1))
    y = np.array(L)
    fd = -np.polyfit(x, y, 1)[0]
    return fd

fd = higuchi_fd(returns.values)
print(f"Fractal Dimension (Higuchi): {fd:.4f}")




def mfdfa(series, q_list=range(-5, 6), max_lag=50):
    """Compute Multifractal Hurst exponents."""
    n = len(series)
    lags = range(10, max_lag)
    F_q = {q: [] for q in q_list}
    
    for lag in lags:
        chunks = [series[i:i + lag] for i in range(0, n, lag)]
        rms = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            x = np.arange(len(chunk))
            y = chunk
            coeffs = np.polyfit(x, y, 1)
            trend = np.polyval(coeffs, x)
            detrended = y - trend
            rms.append(np.mean(detrended**2))
        
        for q in q_list:
            if q == 0:
                F_q[q].append(np.exp(0.5 * np.mean(np.log(rms))))
            else:
                F_q[q].append((np.mean(np.power(rms, q/2))) ** (1/q))
    
    H_q = {}
    for q in q_list:
        if len(F_q[q]) > 0:
            poly = np.polyfit(np.log(lags), np.log(F_q[q]), 1)
            H_q[q] = poly[0]
    return H_q

def predict_direction(prices, window=100, future_steps=5):
    """Predict price direction using fractal trends."""
    returns = np.log(prices).diff().dropna()
    recent_data = returns[-window:].values
    
    # Calculate Hurst exponent (q=2 for standard DFA)
    H_q = mfdfa(recent_data, q_list=[2])
    H = H_q[2]
    
    # Strategy logic
    if H > 0.55:
        # Persistent trend: extrapolate last trend
        X = np.arange(window).reshape(-1, 1)
        y = recent_data
        model = LinearRegression()
        model.fit(X, y)
        future_trend = model.predict([[window + future_steps]])[0]
        direction = "UP" if future_trend > 0 else "DOWN"
    elif H < 0.45:
        # Mean-reverting: predict reversal
        direction = "DOWN" if recent_data[-1] > 0 else "UP"
    else:
        direction = "NEUTRAL (Random)"
    
    return H, direction

# Example usage
if __name__ == "__main__":
    # Load your price data (replace with actual data)
    prices = pd.Series(np.cumsum(np.random.randn(1000))) * 10 + 100  # Synthetic data
    
    # Predict next 5 steps
    H, direction = predict_direction(prices, window=200, future_steps=5)
    print(f"Hurst (q=2): {H:.3f} → Predicted Direction: {direction}")