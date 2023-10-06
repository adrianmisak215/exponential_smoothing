import numpy as np
import pandas as pd

def moving_average(d: list, extra_periods: int = 1, n: int = 3) -> list:
    """
    Performs moving average for extra_periods in the future based on time series
    data in d.
    @input:
        d: time series data
        extra_periods: number of periods in the future to forecast
        n: number of periods to use in the moving average calculation
    @output:
        moving average forecast in a list
    """

    cols = len(d)
    d = np.append(d, [np.nan] * extra_periods)
    f = np.full(cols + extra_periods, np.nan)

    for t in range(n, cols):
        f[t] = np.mean(d[t - n:t])

    f[t+1:] = np.mean(d[t-n+1:t+1])

    df = pd.DataFrame.from_dict({
        "demand": d,
        "forecast": f,
        "error": d - f,
    })

    return df


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


def simple_exp_smooth(d: list, extra_periods: int = 1, alpha: float = 0.4):

    cols = len(d)

    d = np.append(d, [np.nan] * extra_periods)

    f = np.full(cols + extra_periods, np.nan)

    f[1] = d[0]

    for t in range(2, cols + 1):
        f[t] = alpha * d[t - 1] + (1 - alpha) * f[t - 1]

    for t in range(cols + 1, cols + extra_periods):
        f[t] = f[t - 1]

    df = pd.DataFrame.from_dict({
        "demand": d,
        "forecast": f,
        "error": d - f,
    })

    return df


def double_exp_smooth(d: list, extra_periods: int = 1, alpha: float = 0.4, beta: float = 0.4):

    cols = len(d)

    d = np.append(d, [np.nan] * extra_periods)

    f = np.full(cols + extra_periods, np.nan)
    a = np.full(cols + extra_periods, np.nan)
    b = np.full(cols + extra_periods, np.nan)

    a[0] = d[0]
    b[0] = d[1] - d[0]

    # create all t+1 forecasts
    for t in range(1, cols + 1):
        f[t] = a[t - 1] + b[t - 1]
        a[t] = alpha * d[t] + (1 - alpha) * (a[t - 1] + b[t - 1])
        b[t] = beta * (a[t] - a[t - 1]) + (1 - beta) * b[t - 1]

    for t in range(cols, cols + extra_periods):
        f[t] = a[t - 1] + b[t - 1]
        a[t] = f[t]
        b[t] = b[t - 1]

    df = pd.DataFrame({
        "demand": d,
        "forecast": f,
        "level": a,
        "trend": b,
        "error": d - f,
    })

    return df


def exp_smooth_opti(d: list, extra_periods: int = 6):

    params = []
    KPIs = []
    dfs = []

    for alpha in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:

        df = simple_exp_smooth(d, extra_periods=extra_periods, alpha=alpha)
        MAE = df["error"].abs().mean()

        params.append("Simple Smoothing, alpha: {}".format(alpha))
        KPIs.append(MAE)
        dfs.append(df)

        for beta in [0.05, 0.1, 0.2, 0.3, 0.4]:

            df = double_exp_smooth(d, extra_periods=extra_periods, alpha=alpha, beta=beta)
            MAE = df["error"].abs().mean()

            params.append("Double Smoothing, alpha: {}, beta: {}".format(alpha, beta))
            KPIs.append(MAE)
            dfs.append(df)

    mini = np.argmin(KPIs)

    print("Best solution found for {} MAE of {}".format(params[mini], KPIs[mini]))

    return dfs[mini]


def double_exp_smooth_damped(d: list, extra_periods: int = 1, alpha: float = 0.4, beta: float = 0.4, phi: float = 0.9):

    cols = len(d)
    d = np.append(d, [np.nan] * extra_periods)

    f = np.full(cols + extra_periods, np.nan)
    a = np.full(cols + extra_periods, np.nan)
    b = np.full(cols + extra_periods, np.nan)

    a[0] = d[0]
    b[0] = d[1] - d[0]

    # create all t+1 forecasts
    for t in range(1, cols + 1):
        f[t] = a[t - 1] + phi * b[t - 1]
        a[t] = alpha * d[t] + (1 - alpha) * (a[t - 1] + phi * b[t - 1])
        b[t] = beta * (a[t] - a[t - 1]) + (1 - beta) * phi * b[t - 1]

    for t in range(cols, cols + extra_periods):
        f[t] = a[t - 1] + phi * b[t - 1]
        a[t] = f[t]
        b[t] = phi * b[t - 1]

    df = pd.DataFrame({
        "demand": d,
        "forecast": f,
        "level": a,
        "trend": b,
        "error": d - f,
    })

    return df


def seasonal_factors_mul(s, d, slen, cols):
    for i in range(slen):
        s[i] = np.mean(d[i:cols:slen]) / np.mean(s[:slen])
    return s

def triple_exp_smooth_mul(d, slen=12, extra_periods=1, alpha=0.4, beta=0.4, phi=0.9, gamma=0.3):

    cols = len(d)
    d = np.append(d, [np.nan] * extra_periods)

    f = np.full(cols + extra_periods, np.nan)
    a = np.full(cols + extra_periods, np.nan)
    b = np.full(cols + extra_periods, np.nan)
    s = np.full(cols + extra_periods, np.nan)

    s = seasonal_factors_mul(s, d, slen, cols)
    print(s)

    a[0] = d[0] / s[0]
    b[0] = d[1] / s[1] - d[0] / s[0]

    for t in range(1, slen):
        f[t] = (a[t-1] + phi * b[t-1]) * s[t]
        a[t] = alpha*d[t]/s[t] + (1-alpha)*(a[t-1] + phi * b[t-1])
        b[t] = beta*(a[t] - a[t-1]) + (1-beta)*phi*b[t-1]

    for t in range(slen, cols):
        f[t] = (a[t-1] + phi * b[t-1]) * s[t-slen]
        a[t] = alpha*d[t]/s[t-slen] + (1-alpha)*(a[t-1] + phi * b[t-1])
        b[t] = beta*(a[t] - a[t-1]) + (1-beta)*phi*b[t-1]
        s[t] = gamma*d[t]/a[t] + (1-gamma)*s[t-slen]

    for t in range(cols, cols + extra_periods):
        f[t] = (a[t-1] + phi * b[t-1]) * s[t-slen]
        a[t] = f[t] / s[t-slen]
        b[t] = phi * b[t-1]
        s[t] = s[t-slen]

    df = pd.DataFrame({
        "demand": d,
        "forecast": f,
        "level": a,
        "trend": b,
        "season": s,
        "error": d - f,
    })

    return df