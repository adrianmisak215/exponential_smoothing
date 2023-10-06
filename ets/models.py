import numpy as np
import pandas as pd

def moving_average(time_series: list, horizon: int, n: int) -> pd.DataFrame:
    """
    Receives the list containing time series data, and uses the moving
    average (window defined by parameter n) to forecast horizon periods into the future.
    @input:
        time_series: list, time series data
        horizon: int, number of periods to forecast into the future
        n: int, window size for moving average
    @output:
        list
    """

    number_historical_data = len(time_series)

    demand = np.append(time_series, [np.nan] * horizon)
    forecast = np.full(number_historical_data + horizon, np.nan)

    for ind in range(n, number_historical_data):
        forecast[ind] = np.mean(demand[ind - n:ind])

    forecast[ind+1:] = np.mean(demand[ind - n + 1:ind + 1])

    return pd.DataFrame(
        {
            'demand': demand, 
            'forecast': forecast,
            'error': demand - forecast
        }
    )


def kpi(df) -> None:
    """
    Calculates the bias, and scaled bias for the forecasted values in the df.
    """

    dem_ave = df.loc[df["error"].notnull(), "demand"].mean()
    bias_abs = df["error"].mean()
    bias_rel = bias_abs / dem_ave
    print("Bias: {:0.2f}, {:.2%}".format(bias_abs, bias_rel))
    MAPE = (df["error"].abs() / df["demand"]).mean()
    print("MAPE: {:.2%}".format(MAPE))
    MAE_abs = df["error"].abs().mean()
    MAE_rel = MAE_abs / dem_ave
    print("MAE: {:0.2f}, {:.2%}".format(MAE_abs, MAE_rel))
    RMSE_abs = np.sqrt((df["error"] ** 2).mean())
    RMSE_rel = RMSE_abs / dem_ave
    print("RMSE: {:0.2f}, {:.2%}".format(RMSE_abs, RMSE_rel))


def simple_exponential_smoothing(time_series: list, horizon: int, alpha: float = 0.4) -> pd.DataFrame:
    """
    Implements the simple exponential smoothing method to forecast horizon periods into the future.
    @input:
        time_series: list, time series data
        horizon: int, number of periods to forecast into the future
        alpha: float, smoothing parameter
    @output:
        forecast pd.DataFrame
    """

    number_historical_data = len(time_series)

    demand = np.append(time_series, [np.nan] * horizon)
    forecast = np.full(number_historical_data + horizon, np.nan)

    forecast[1] = demand[0]

    for ind in range(2, number_historical_data + 1):
        forecast[ind] = alpha * demand[ind - 1] + (1 - alpha) * forecast[ind - 1]

    forecast[ind+1:] = forecast[ind]

    return pd.DataFrame(
        {
            'demand': demand, 
            'forecast': forecast,
            'error': demand - forecast
        }
    )


def double_exponential_smoothing(time_series: list, horizon: int, alpha: float = 0.4, beta: float = 0.4) -> pd.DataFrame:
    """
    Implements the double exponential smoothing, which introduces the trend component, to forecast horizon periods into the future.
    @input:
        time_series: list, time series data
        horizon: int, number of periods to forecast into the future
        alpha: float, smoothing parameter for level
        beta: float, smoothing parameter for trend
    @output:
        forecast pd.DataFrame
    """

    number_historical_data = len(time_series)

    demand = np.append(time_series, [np.nan] * horizon)

    f = np.full(number_historical_data + horizon, np.nan)
    a = np.full(number_historical_data + horizon, np.nan)
    b = np.full(number_historical_data + horizon, np.nan)

    a[0] = demand[0]
    b[0] = demand[1] - demand[0]

    for ind in range(1, number_historical_data + 1):
        f[ind] = a[ind - 1] + b[ind - 1]
        a[ind] = alpha * demand[ind] + (1 - alpha) * (a[ind - 1] + b[ind - 1])
        b[ind] = beta * (a[ind] - a[ind - 1]) + (1 - beta) * b[ind - 1]

    for ind in range(number_historical_data, number_historical_data + horizon):
        f[ind] = a[ind - 1] + b[ind - 1]
        a[ind] = f[ind]
        b[ind] = b[ind - 1]

    return pd.DataFrame(
        {
            'demand': demand, 
            'forecast': f,
            "level": a,
            "trend": b,
            'error': demand - f
        }
    )


def double_exponential_smoothing_damped_trend(time_series: list, horizon: int, alpha: float = 0.4, beta: float = 0.4, phi: float = 0.9) -> pd.DataFrame:
    """
    Implements the double exponential smoothing, with damped trend to forecast horizon periods into the future.
    @input:
        time_series: list, time series data
        horizon: int, number of periods to forecast into the future
        alpha: float, smoothing parameter for level
        beta: float, smoothing parameter for trend
        phi: float, damping parameter for trend
    @output:
        forecast pd.DataFrame
    """

    number_historical_data = len(time_series)

    demand = np.append(time_series, [np.nan] * horizon)

    f = np.full(number_historical_data + horizon, np.nan)
    a = np.full(number_historical_data + horizon, np.nan)
    b = np.full(number_historical_data + horizon, np.nan)

    a[0] = demand[0]
    b[0] = demand[1] - demand[0]

    for ind in range(1, number_historical_data + 1):
        f[ind] = a[ind - 1] + phi * b[ind - 1]
        a[ind] = alpha * demand[ind] + (1 - alpha) * (a[ind - 1] + phi * b[ind - 1])
        b[ind] = beta * (a[ind] - a[ind - 1]) + (1 - beta) * phi * b[ind - 1]

    for ind in range(number_historical_data, number_historical_data + horizon):
        f[ind] = a[ind - 1] + phi * b[ind - 1]
        a[ind] = f[ind]
        b[ind] = phi * b[ind - 1]

    return pd.DataFrame(
        {
            'demand': demand, 
            'forecast': f,
            "level": a,
            "trend": b,
            'error': demand - f
        }
    )


def initialize_seasonal_factors(time_series: list, s: np.array, period_length: int) -> np.array:
    """
    Uses the simple historical method to estimate the seasonal factors.
    The parameters s contains the newly initialized list which will contain seasonal factors, but is
    initially filled with np.nan values. This function will fill the first period_length values of s.
    """
    for i in range(period_length):
        s[i] = np.mean(time_series[i::period_length])
    s /= np.mean(s[:period_length])
    return s


def triple_exponential_smoothing(time_series: list, period_length: int = 12, horizon: int = 1, alpha: float = 0.4, beta: float = 0.4, phi: float = 0.9, gamma: float = 0.3):
    """
    Implements the triple exponential smoothing method to forecast horizon steps ahead.
    @input:
        time_series: list of time series values
        period_length: length of the seasonal period
        horizon: number of steps ahead to forecast
        alpha: smoothing parameter for the level
        beta: smoothing parameter for the trend
        phi: smoothing parameter for the seasonality
        gamma: smoothing parameter for the seasonal trend
    @output:
        forecasts: list of forecasts
    """

    number_historical_data = len(time_series)

    demand = np.append(time_series, np.full(horizon, np.nan))
    f = np.full(number_historical_data + horizon, np.nan)
    a = np.full(number_historical_data + horizon, np.nan)
    b = np.full(number_historical_data + horizon, np.nan)
    s = np.full(number_historical_data + horizon, np.nan)

    s = initialize_seasonal_factors(time_series, s, period_length)

    a[0] = demand[0] / s[0]
    b[0] = demand[1] / s[1] - demand[0] / s[0]

    for t in range(1, period_length):
        f[t] = (a[t-1] + phi*b[t-1]) * s[t]
        a[t] = alpha * demand[t] / s[t] + (1-alpha) * (a[t-1] + phi * b[t-1])
        b[t] = beta * (a[t] - a[t-1]) + (1-beta)*phi*b[t-1]

    for t in range(period_length, number_historical_data):
        f[t] = (a[t-1] + phi*b[t-1]) * s[t-period_length]
        a[t] = alpha * demand[t] / s[t-period_length] + (1-alpha) * (a[t-1] + phi * b[t-1])
        b[t] = beta * (a[t] - a[t-1]) + (1-beta)*phi*b[t-1]
        s[t] = gamma * demand[t] / a[t] + (1-gamma) * s[t-period_length]

    for t in range(number_historical_data, number_historical_data + horizon):
        f[t] = (a[t-1] + phi*b[t-1]) * s[t-period_length]
        a[t] = f[t] / s[t-period_length]
        b[t] = phi * b[t-1]
        s[t] = s[t-period_length]

    return pd.DataFrame({
        "demand": demand, "forecast": f, "level": a, "trend": b, "season": s, "error": demand - f
    })